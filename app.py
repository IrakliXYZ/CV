from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, after_this_request
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import uuid
import anthropic
import PyPDF2
import io
from datetime import datetime
import html
import json
import logging
from dotenv import load_dotenv
import tempfile
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Set up Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

logger.info(f"Flask app configured with upload folder: {app.config['UPLOAD_FOLDER']}")

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

# Create session folder if it doesn't exist
if not os.path.exists(app.config['SESSION_FILE_DIR']):
    os.makedirs(app.config['SESSION_FILE_DIR'])
    logger.info(f"Created session folder: {app.config['SESSION_FILE_DIR']}")

# Set up additional directories
app.config['EXPORT_DIR'] = os.path.join(os.getcwd(), 'exports')
if not os.path.exists(app.config['EXPORT_DIR']):
    os.makedirs(app.config['EXPORT_DIR'])
    logger.info(f"Created exports folder: {app.config['EXPORT_DIR']}")

# Initialize Claude client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
logger.info("Claude client initialized")

def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    logger.debug(f"File {filename} allowed: {is_allowed}")
    return is_allowed

def extract_text_from_pdf(file_stream):
    """Extract text content from a PDF file"""
    logger.info("Starting PDF text extraction")
    try:
        reader = PyPDF2.PdfReader(file_stream)
        logger.debug(f"PDF has {len(reader.pages)} pages")
        text = ""
        for page_num in range(len(reader.pages)):
            logger.debug(f"Extracting text from page {page_num+1}")
            text += reader.pages[page_num].extract_text()
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return f"Error extracting text from PDF: {str(e)}"

def chunk_cv_text(cv_text, max_chunk_size=8000):
    """Split CV text into manageable chunks for API calls"""
    logger.info(f"Chunking CV text of length {len(cv_text)} characters")
    
    # If CV is small enough, return as single chunk
    if len(cv_text) <= max_chunk_size:
        logger.debug("CV text is small enough to process in one chunk")
        return [cv_text]
    
    chunks = []
    # Try to split on logical boundaries (double newlines often indicate section breaks)
    paragraphs = cv_text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, start a new chunk
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we couldn't find good paragraph breaks, fall back to character-based chunking
    if not chunks or any(len(chunk) > max_chunk_size for chunk in chunks):
        logger.debug("Falling back to character-based chunking")
        chunks = []
        for i in range(0, len(cv_text), max_chunk_size):
            chunks.append(cv_text[i:i+max_chunk_size])
    
    logger.info(f"Split CV into {len(chunks)} chunks")
    return chunks

def merge_cv_data(cv_data_list):
    """Merge multiple CV data dictionaries into one comprehensive result"""
    logger.info(f"Merging {len(cv_data_list)} CV data chunks")
    
    if not cv_data_list:
        logger.error("No CV data chunks to merge")
        return None
    
    # Start with the first chunk as our base
    result = cv_data_list[0]
    
    # List of fields that should be merged as lists (avoiding duplicates)
    list_fields = ["education", "countries", "languages", "employment", "experience", "tasks"]
    
    for cv_data in cv_data_list[1:]:
        if not cv_data:
            continue
            
        # For each field in the subsequent chunks
        for field, value in cv_data.items():
            # If it's a list field, extend the base list with new unique items
            if field in list_fields:
                if isinstance(value, list):
                    # For each list item in the value
                    for item in value:
                        # Check if this item is already in the result list
                        if field == "education":
                            if not any(e.get("institution") == item.get("institution") and 
                                      e.get("degree") == item.get("degree") for e in result.get(field, [])):
                                result.setdefault(field, []).append(item)
                        elif field == "employment":
                            if not any(e.get("employer") == item.get("employer") and 
                                      e.get("from") == item.get("from") for e in result.get(field, [])):
                                result.setdefault(field, []).append(item)
                        elif field == "experience":
                            if not any(e.get("project_name") == item.get("project_name") and 
                                      e.get("year") == item.get("year") for e in result.get(field, [])):
                                result.setdefault(field, []).append(item)
                        elif field in ["countries", "languages"]:
                            if item not in result.get(field, []):
                                result.setdefault(field, []).append(item)
                        else:
                            result.setdefault(field, []).append(item)
            # For scalar fields, prefer non-empty values
            elif field not in result or not result[field]:
                result[field] = value
            # For world_bank_experience_details, combine if there's new information
            elif field == "world_bank_experience_details" and value and value != "N/A" and value != result[field]:
                result[field] = result[field] + "\n\n" + value
    
    logger.info("Successfully merged CV data chunks")
    return result

def split_cv_by_sections(cv_text):
    """Split CV into basic info section and work undertaken section based on typical CV structure"""
    logger.info("Splitting CV into logical sections")
    
    # Common patterns that might indicate the start of work experience section
    work_section_indicators = [
        "WORK UNDERTAKEN", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", 
        "EMPLOYMENT HISTORY", "RELEVANT EXPERIENCE", "PROJECT EXPERIENCE",
        "ASSIGNMENTS", "KEY PROJECTS", "RELEVANT PROJECTS", "SELECTED PROJECTS",
        "PROJECT HISTORY", "PROFESSIONAL HISTORY", "WORK HISTORY"
    ]
    
    # Try to find the work section in the CV text
    work_section_start = -1
    for indicator in work_section_indicators:
        index = cv_text.upper().find(indicator)
        if index != -1:
            logger.debug(f"Found work section indicator '{indicator}' at position {index}")
            work_section_start = index
            break
    
    # If we found a work section, split the CV
    if work_section_start > 0:
        # Get some context before the exact match point (usually there's a heading, etc.)
        context_start = max(0, work_section_start - 200)
        
        # Find a clean paragraph break before the work section
        for i in range(work_section_start, context_start, -1):
            if i > 0 and cv_text[i-1:i+1] == "\n\n":
                work_section_start = i
                break
        
        basic_info = cv_text[:work_section_start].strip()
        work_section = cv_text[work_section_start:].strip()
        
        logger.info(f"Successfully split CV into basic info ({len(basic_info)} chars) and work section ({len(work_section)} chars)")
        return basic_info, work_section
    else:
        # If we couldn't identify the work section, just return the whole CV
        logger.warning("Could not identify work section in CV, treating as a single section")
        return cv_text, ""

def chunk_work_section(work_section, max_chunk_size=8000):
    """Split the work section into manageable chunks based on project entries"""
    logger.info(f"Chunking work section of length {len(work_section)}")
    
    if not work_section or len(work_section) <= max_chunk_size:
        return [work_section] if work_section else []
    
    # Common patterns that might indicate the start of a new project
    project_indicators = [
        "PROJECT NAME", "NAME OF ASSIGNMENT", "NAME OF PROJECT",
        "PROJECT TITLE", "ASSIGNMENT TITLE", "CLIENT:", "CLIENT :",
        "EMPLOYER:", "EMPLOYER :", "DURATION:", "DURATION :", 
        "PERIOD:", "PERIOD :", "YEAR:", "YEAR :"
    ]
    
    # Find potential project starting points
    project_starts = []
    for indicator in project_indicators:
        # Find all occurrences of this indicator
        for match in re.finditer(indicator, work_section, re.IGNORECASE):
            # Get the position of the start of the line containing this indicator
            line_start = work_section.rfind('\n', 0, match.start())
            if line_start == -1:  # If it's on the first line
                line_start = 0
            else:
                line_start += 1  # Skip the newline character
            
            project_starts.append(line_start)
    
    # Sort and deduplicate project starting positions
    project_starts = sorted(set(project_starts))
    
    # If we couldn't identify project boundaries, fall back to regular chunking
    if len(project_starts) <= 1:
        logger.warning("Could not identify project boundaries in work section, using regular chunking")
        return chunk_cv_text(work_section, max_chunk_size)
    
    # Create chunks based on project boundaries
    chunks = []
    current_chunk = ""
    
    # Add the introduction part to the first chunk
    if project_starts[0] > 0:
        current_chunk = work_section[:project_starts[0]]
    
    for i in range(len(project_starts)):
        start = project_starts[i]
        end = project_starts[i+1] if i+1 < len(project_starts) else len(work_section)
        project_text = work_section[start:end]
        
        # If adding this project would exceed max size, start a new chunk
        if len(current_chunk) + len(project_text) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = project_text
        else:
            current_chunk += project_text
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Split work section into {len(chunks)} chunks based on project boundaries")
    return chunks

def extract_year_for_sorting(year_str):
    """Extract a sortable year value from various date formats.
    Handles formats like "2020", "2019-2020", "Jan 2020", "2020-present", "Jan 2019 - Dec 2020", etc.
    Will try to extract the most recent year if date range is provided.
    """
    logger.debug(f"Extracting sortable year from: {year_str}")
    if not year_str or not isinstance(year_str, str):
        return 0
    
    # Try to extract all years (4-digit numbers) from the string
    import re
    years = re.findall(r'(?:19|20)\d{2}', year_str)
    
    if years:
        # Check for words indicating "present" or "ongoing"
        if any(word in year_str.lower() for word in ["present", "current", "ongoing", "now", "to date"]):
            # If "present" is indicated, use current year as the most recent
            return int(datetime.now().year)
        
        # Otherwise, get the largest (most recent) year
        return max(int(year) for year in years)
    
    # If no 4-digit years found, check for abbreviated years (e.g., "'20" for 2020)
    abbr_years = re.findall(r"'(\d{2})", year_str)
    if abbr_years:
        # Convert '19 to 2019, '20 to 2020, etc.
        full_years = [2000 + int(year) if int(year) < 50 else 1900 + int(year) for year in abbr_years]
        return max(full_years)
    
    # If no years found, return 0 (will sort to the end)
    return 0

def sort_experiences(experiences):
    """Sort experiences from most recent to oldest based on year information"""
    logger.info(f"Sorting {len(experiences)} work experiences by year")
    
    # First try to extract and sort by the most recent year mentioned
    return sorted(
        experiences,
        key=lambda x: extract_year_for_sorting(str(x.get("year", ""))),
        reverse=True  # Most recent first
    )

def get_cv_data_from_claude(cv_text, position, tasks, wb_experience, cv_format="world_bank"):
    """Use Claude to extract structured CV data, handling large documents by chunking based on CV structure"""
    logger.info(f"Getting CV data from Claude for format: {cv_format}")
    format_name_map = {
        "world_bank": "World Bank",
        "adb": "ADB (Asian Development Bank)",
        "ebrd": "EBRD (European Bank for Reconstruction and Development)",
        "undp": "UNDP (United Nations Development Programme)"
    }
    
    format_name = format_name_map.get(cv_format, "World Bank")
    logger.debug(f"Using format name: {format_name}")
    
    # Break tasks into a list
    task_list = [task.strip() for task in tasks.split(',') if task.strip()]
    logger.debug(f"Parsed {len(task_list)} tasks")
    
    # Split CV into basic info and work experience sections
    basic_info, work_section = split_cv_by_sections(cv_text)
    
    # If we couldn't split, just process the whole CV as before
    if not work_section:
        logger.info("Processing CV as a single document (no section split)")
        if len(cv_text) > 10000:
            return process_cv_in_chunks(cv_text, position, task_list, wb_experience, format_name)
        else:
            return process_cv_chunk(cv_text, position, task_list, wb_experience, format_name)
    
    # Process the basic info section first
    logger.info("Processing basic information section")
    basic_info_data = process_cv_chunk(
        basic_info, 
        position, 
        task_list, 
        wb_experience, 
        format_name,
        section_type="basic_info"
    )
    
    if not basic_info_data:
        logger.error("Failed to process basic information section")
        flash("Failed to extract basic information from CV", "error")
        return None
    
    # Now process the work experience section in chunks if needed
    if len(work_section) > 10000:
        logger.info("Work section is large, processing in chunks")
        work_chunks = chunk_work_section(work_section)
        
        # Process each work chunk and collect results
        work_data_list = []
        for i, chunk in enumerate(work_chunks):
            logger.info(f"Processing work experience chunk {i+1}/{len(work_chunks)}")
            chunk_data = process_cv_chunk(
                chunk,
                position,
                task_list,
                wb_experience,
                format_name,
                section_type="work_experience",
                chunk_num=i+1,
                total_chunks=len(work_chunks)
            )
            if chunk_data:
                work_data_list.append(chunk_data)
        
        # Merge all experience data
        if not work_data_list:
            logger.error("Failed to process any work experience chunks")
            # Continue with just basic info
        else:
            logger.info(f"Successfully processed {len(work_data_list)} work experience chunks")
            
            # Extract just the experience section from each chunk and combine
            all_experience = []
            for data in work_data_list:
                if "experience" in data and isinstance(data["experience"], list):
                    all_experience.extend(data["experience"])
            
            # Sort experiences by year from most recent to oldest
            sorted_experience = sort_experiences(all_experience)
            logger.info(f"Sorted {len(sorted_experience)} experiences by year")
            
            # Update the base data with the combined experience
            basic_info_data["experience"] = sorted_experience
    else:
        # Process work section as a single chunk
        logger.info("Processing work experience section as a single chunk")
        work_data = process_cv_chunk(
            work_section,
            position,
            task_list,
            wb_experience,
            format_name,
            section_type="work_experience"
        )
        
        if work_data and "experience" in work_data and isinstance(work_data["experience"], list):
            # Sort experiences by year from most recent to oldest
            sorted_experience = sort_experiences(work_data["experience"])
            basic_info_data["experience"] = sorted_experience
    
    logger.info("CV data extraction complete")
    return basic_info_data

def process_cv_in_chunks(cv_text, position, task_list, wb_experience, format_name):
    """Process CV in standard chunks when section splitting isn't possible"""
    logger.info("Processing CV in standard chunks")
    
    cv_chunks = chunk_cv_text(cv_text, max_chunk_size=10000)
    logger.info(f"CV text split into {len(cv_chunks)} chunks")
    
    if len(cv_chunks) == 1:
        logger.info("Processing CV as a single chunk")
        return process_cv_chunk(cv_chunks[0], position, task_list, wb_experience, format_name)
    else:
        logger.info(f"Processing CV in {len(cv_chunks)} chunks")
        chunk_results = []
        
        # Process each chunk and collect results
        for i, chunk in enumerate(cv_chunks):
            logger.info(f"Processing chunk {i+1}/{len(cv_chunks)}")
            chunk_result = process_cv_chunk(
                chunk, 
                position, 
                task_list, 
                wb_experience, 
                format_name, 
                is_chunk=True, 
                chunk_num=i+1, 
                total_chunks=len(cv_chunks)
            )
            if chunk_result:
                chunk_results.append(chunk_result)
        
        # Merge results from all chunks
        if chunk_results:
            logger.info(f"Successfully processed {len(chunk_results)} CV chunks")
            return merge_cv_data(chunk_results)
        else:
            logger.error("All chunks failed to process")
            flash("Failed to process CV with Claude AI", "error")
            return None

def process_cv_chunk(cv_chunk, position, task_list, wb_experience, format_name, section_type=None, is_chunk=False, chunk_num=1, total_chunks=1):
    """Process a single chunk of the CV text using Claude API"""
    chunk_info = f" (Part {chunk_num}/{total_chunks})" if is_chunk else ""
    section_info = f" ({section_type})" if section_type else ""
    logger.info(f"Processing CV chunk{chunk_info}{section_info}")
    
    # Modify prompt based on chunk type and section
    chunk_instruction = ""
    if is_chunk:
        chunk_instruction = f"""
This is part {chunk_num} of {total_chunks} of the CV. 
Focus on extracting information present in this chunk only. 
Don't worry if some sections appear incomplete - I'll combine data from all chunks later.
"""
    elif section_type == "basic_info":
        chunk_instruction = """
This is the first part of the CV containing basic personal information, education, languages, etc.
Focus on extracting all information EXCEPT for the detailed work experience projects.
For the "experience" field, leave it as an empty array - we'll process that separately.
"""
    elif section_type == "work_experience":
        chunk_instruction = """
This section contains the "Work Undertaken that Best Illustrates Capability to Handle the Tasks Assigned" part of the CV.
Focus ONLY on extracting the detailed project experiences for the "experience" array.
For all other fields (name, education, etc.), you can leave them empty as we already have that information.
"""
    
    prompt = f"""
I need you to extract information from this CV/resume to format it according to the {format_name} template.
{chunk_instruction}
Here's the CV content:

{cv_chunk}

The proposed position is: {position}

The detailed tasks are:
{", ".join(task_list)}

World Bank experience information: {wb_experience}

Please extract all the required information in JSON format with the following structure:
{{
    "name": "",
    "proposed_position": "{position}",
    "current_employer": "",
    "dob": "",
    "nationality": "",
    "education": [
        {{"institution": "", "degree": "", "date": ""}}
    ],
    "professional_memberships": "",
    "other_training": "",
    "countries": [],
    "languages": [],
    "employment": [
        {{"from": "", "to": "", "employer": "", "positions": ""}}
    ],
    "tasks": {json.dumps(task_list)},
    "experience": [
        {{
            "project_name": "",
            "year": "",
            "location": "",
            "client": "",
            "position_held": "",
            "activities": ""
        }}
    ],
    "world_bank_experience_details": "{wb_experience}"
}}

Extract as many entries as possible for education, countries, languages, employment, and experience sections. Do not include any other text in the response. Do not omit any information.
Format dates as MM/YYYY if possible. If information is not available, leave the field empty or use "N/A".
"""
    logger.debug(f"Created prompt for Claude with {len(prompt)} characters")

    try:
        logger.info("Sending request to Claude API with streaming enabled")
        # Use streaming for potentially long requests
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=64000,
            temperature=0.1,
            system="You are a helpful assistant that extracts structured CV information.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            # Collect the complete response
            content = ""
            for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == "content_block_delta" and hasattr(chunk, 'delta') and chunk.delta.type == "text":
                    content += chunk.delta.text
                elif hasattr(chunk, 'type') and chunk.type == "content_block_start" and hasattr(chunk, 'content_block') and chunk.content_block.type == "text":
                    # Some streams deliver full content blocks instead of deltas
                    content += chunk.content_block.text
            
            # If we didn't get any content from streaming, try getting it from the final message
            if not content:
                final_message = stream.get_final_message()
                if final_message and hasattr(final_message, 'content'):
                    for block in final_message.content:
                        if block.type == "text":
                            content += block.text
            
            logger.info(f"Received streaming response from Claude API")
            logger.info(f"Final content length: {len(content)}")
            logger.debug(f"First 100 chars of content: {content[:100]}")
        
        # Find JSON in the response
        logger.debug(f"Checking for JSON in content...")
        
        # First check for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            logger.debug("Found JSON in code block format")
            json_str = json_match.group(1).strip()
        else:
            # Look for a JSON object that spans most of the content
            json_pattern = r'(\{[\s\S]*\})'
            match = re.search(json_pattern, content)
            if match:
                logger.debug("Found JSON pattern in content")
                json_str = match.group(1).strip()
            else:
                logger.debug("No JSON pattern found, using full response as a last resort")
                json_str = content.strip()
        
        logger.debug(f"Extracted JSON string length: {len(json_str)}")
        logger.debug(f"First 100 chars of JSON string: {json_str[:100]}")
        
        # Clean up any remaining markdown or text
        try:
            cv_data = json.loads(json_str)
            logger.info("Successfully parsed JSON data from Claude response")
            logger.debug(f"Parsed data: {json.dumps(cv_data, indent=2)}")
            return cv_data
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError: {str(e)}")
            logger.warning("Trying advanced JSON extraction techniques")
            
            # If direct parsing fails, try more aggressive cleaning
            # 1. Try to fix unescaped quotes or control characters
            try:
                # Replace common issues
                cleaned_json = json_str.replace('\n', ' ').replace('\r', ' ')
                # Fix trailing commas before closing braces
                cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                
                cv_data = json.loads(cleaned_json)
                logger.info("Successfully parsed JSON after cleaning")
                return cv_data
            except json.JSONDecodeError:
                logger.warning("Cleaning attempt failed")
                
                # 2. Try to extract just a valid JSON object anywhere in the text
                try:
                    # Sometimes the model doesn't format the JSON well, but includes valid JSON within text
                    all_json_objects = re.findall(r'(\{[\s\S]*?\})', json_str)
                    for obj in all_json_objects:
                        try:
                            cv_data = json.loads(obj)
                            if isinstance(cv_data, dict) and (section_type != "work_experience" or "experience" in cv_data):
                                logger.info("Found valid JSON object within text")
                                return cv_data
                        except:
                            pass
                    
                    logger.error("No valid JSON objects found in content")
                    if is_chunk or section_type == "work_experience":
                        # For chunks, don't show flash messages for each failure
                        logger.warning(f"Could not extract structured data from chunk {chunk_num}")
                        return None
                    else:
                        flash("Could not extract structured data from Claude's response", "error")
                        return None
                except Exception as e:
                    logger.error(f"Error during advanced JSON extraction: {str(e)}", exc_info=True)
                    if not is_chunk and section_type != "work_experience":
                        flash("Error parsing Claude's response", "error")
                    return None
    
    except Exception as e:
        logger.error(f"Error communicating with Claude API: {str(e)}", exc_info=True)
        if not is_chunk and section_type != "work_experience":
            flash(f"Error communicating with Claude API: {str(e)}", "error")
        return None

def generate_html_from_template(cv_data):
    """Generate HTML from the template using the extracted CV data"""
    logger.info("Generating HTML from template")
    
    try:
        # Prepare template context
        context = {
            "name": cv_data.get("name", ""),
            "proposed_position": cv_data.get("proposed_position", ""),
            "current_employer": cv_data.get("current_employer", ""),
            "dob": cv_data.get("dob", ""),
            "nationality": cv_data.get("nationality", ""),
            "education": cv_data.get("education", []),
            "professional_memberships": cv_data.get("professional_memberships", ""),
            "other_training": cv_data.get("other_training", ""),
            "countries": cv_data.get("countries", []),
            "languages": cv_data.get("languages", []),
            "employment": cv_data.get("employment", []),
            "tasks": cv_data.get("tasks", []),
            "experience": cv_data.get("experience", []),
            "world_bank_experience_details": cv_data.get("world_bank_experience_details", "N/A"),
            "todays_date": datetime.now().strftime("%d/%m/%Y")
        }
        
        # Render template with context
        result = render_template('cv_template.html', **context)
        logger.info(f"Generated HTML result with {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Error generating HTML from template: {str(e)}", exc_info=True)
        return f"Error generating HTML from template: {str(e)}"

@app.route('/')
def index():
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    logger.info("Upload endpoint called")
    if 'cv_file' not in request.files:
        logger.warning("No file part in request")
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['cv_file']
    if file.filename == '':
        logger.warning("No selected file")
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    logger.info(f"Processing file: {file.filename}")
    if file and allowed_file(file.filename):
        try:
            # Get form data
            cv_format = request.form.get('cv_format', 'world_bank')
            proposed_position = request.form.get('proposed_position', '')
            detailed_tasks = request.form.get('detailed_tasks', '')
            world_bank_experience = request.form.get('world_bank_experience', 'N/A')
            
            logger.info(f"Form data: format={cv_format}, position={proposed_position}")
            logger.debug(f"Tasks: {detailed_tasks}")
            logger.debug(f"WB Experience: {world_bank_experience}")
            
            # Extract text from PDF
            cv_text = extract_text_from_pdf(file)
            logger.debug(f"Extracted {len(cv_text)} characters from PDF")
            
            # Get structured data from Claude
            cv_data = get_cv_data_from_claude(
                cv_text, 
                proposed_position, 
                detailed_tasks, 
                world_bank_experience,
                cv_format
            )
            
            if cv_data:
                logger.info("Successfully got structured data from Claude")
                
                # Generate HTML from template (now directly using Jinja2)
                html_result = generate_html_from_template(cv_data)
                
                # Store data in server-side session
                session['html_result'] = html_result
                session['cv_format'] = cv_format
                
                # Store the person's name in the session for filenames
                if cv_data and 'name' in cv_data:
                    session['person_name'] = cv_data['name']
                else:
                    session['person_name'] = 'candidate'
                
                logger.info("Stored results in session")
                
                format_name_map = {
                    "world_bank": "World Bank",
                    "adb": "ADB (Asian Development Bank)",
                    "ebrd": "EBRD (European Bank for Reconstruction and Development)",
                    "undp": "UNDP (United Nations Development Programme)"
                }
                format_name = format_name_map.get(cv_format, "World Bank")
                
                logger.info(f"Rendering result page for format: {format_name}")
                
                # Send the HTML result and format name to the result template
                return render_template('result.html', html_result=html_result, format_name=format_name)
                
            else:
                logger.error("Failed to process CV with Claude AI")
                flash('Failed to process CV with Claude AI', 'error')
                return redirect(url_for('index'))
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        logger.warning(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

@app.route('/download_html')
def download_html():
    logger.info("Download HTML endpoint called")
    html_result = session.get('html_result')
    cv_format = session.get('cv_format', 'world_bank')
    person_name = session.get('person_name', 'candidate')
    
    # Sanitize the name for file naming (replace spaces with underscores)
    safe_name = person_name.replace(' ', '_')
    
    if not html_result:
        logger.warning("No CV content found in session")
        flash('No CV content found in session', 'error')
        return redirect(url_for('index'))
    
    format_name_map = {
        "world_bank": "World_Bank",
        "adb": "ADB",
        "ebrd": "EBRD",
        "undp": "UNDP"
    }
    format_name = format_name_map.get(cv_format, "World_Bank")
    
    # Create a BytesIO object
    html_io = io.BytesIO()
    html_io.write(html_result.encode('utf-8'))
    html_io.seek(0)
    
    logger.info(f"Sending HTML file with name: {safe_name}_{format_name}_CV.html")
    return send_file(
        html_io,
        mimetype='text/html',
        as_attachment=True,
        download_name=f'{safe_name}_{format_name}_CV.html'
    )

@app.route('/export')
def export():
    logger.info("Export endpoint called")
    format_type = request.args.get('format', 'html')
    logger.debug(f"Export format: {format_type}")
    
    html_result = session.get('html_result')
    cv_format = session.get('cv_format', 'world_bank')
    person_name = session.get('person_name', 'candidate')
    
    # Sanitize the name for file naming (replace spaces with underscores)
    safe_name = person_name.replace(' ', '_')
    
    format_name_map = {
        "world_bank": "World_Bank",
        "adb": "ADB",
        "ebrd": "EBRD",
        "undp": "UNDP"
    }
    format_name = format_name_map.get(cv_format, "World_Bank")
    
    if not html_result:
        logger.warning("No CV content found in session")
        flash('No CV content found in session', 'error')
        return redirect(url_for('index'))
    
    if format_type == 'html':
        logger.info("Exporting as HTML")
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = os.path.join(app.config['EXPORT_DIR'], f"{safe_name}_{format_name}_CV_{timestamp}.html")
            
            # Write HTML to file
            with open(html_path, 'w', encoding='utf-8') as html_file:
                html_file.write(html_result)
            
            # Generate a friendly filename for the browser
            download_filename = f"{safe_name}_{format_name}_CV.html"
            logger.info(f"Sending HTML file with name: {download_filename}")
            
            # Send the file
            return send_file(
                html_path,
                mimetype='text/html',
                as_attachment=True,
                download_name=download_filename
            )
        except Exception as e:
            logger.error(f"Error exporting HTML: {str(e)}", exc_info=True)
            flash(f'Error exporting HTML: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        logger.warning(f"Unsupported export format: {format_type}")
        flash('Unsupported export format', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5005)