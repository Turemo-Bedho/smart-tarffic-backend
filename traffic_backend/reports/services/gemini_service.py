
# import json
# import logging
# from django.conf import settings
# import google.generativeai as genai
# from datetime import datetime, timedelta
# import re

# logger = logging.getLogger(__name__)

# class GeminiTrafficParser:
#     def __init__(self):
#         if not hasattr(settings, 'GEMINI_API_KEY'):
#             raise ValueError("GEMINI_API_KEY missing in Django settings")

#         genai.configure(api_key=settings.GEMINI_API_KEY)
#         self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
#         self.safety_settings = {
#             'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
#             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
#             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
#             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
#         }

#     def _get_schema_prompt(self) -> str:
#         """Returns formatted database schema for Gemini."""
#         return """
#         Database Schema(Table → Fields):
#         - driver_api_driver: id, license_number, first_name, last_name, sex, phone_number,nationality, license_issue_date,
#         - driver_api_vehicle: license_plate, make, model, year, color, registration_date
#         - driver_api_violation: driver_id, vehicle_id, violation_type_id, issued_by_officer_id, location, created_at
#         - driver_api_ticket: violation_id, status (PENDING/PAID/FAILED), reference_code, issued_at
#         - driver_api_officer: user_id, badge_number
#         - driver_api_address: id, street, city, state, zip_code
#         -driver_api_violationtype: id, name, description
#         """

#     def _standardize_entities(self, entities: dict) -> dict:
#         """Normalizes extracted entities to prevent processing errors."""
#         if 'license_number' in entities and entities['license_number']:
#             entities['license_number'] = re.sub(r'[^A-Z0-9]', '', str(entities['license_number'])).upper()
#         else:
#             entities['license_number'] = None

#         if 'license_plate' in entities and entities['license_plate']:
#             entities['license_plate'] = re.sub(r'[^A-Z0-9]', '', str(entities['license_plate'])).upper()
#         else:
#             entities['license_plate'] = None

#         violation_mappings = {
#             'speed': 'SPEEDING',
#             'helmet': 'NO_HELMET',
#             'light': 'RED_LIGHT_VIOLATION'
#         }
#         if 'violation_type' in entities and entities['violation_type']:
#             v_type = entities['violation_type'].lower()
#             entities['violation_type'] = violation_mappings.get(v_type, v_type.upper())
#         else:
#             entities['violation_type'] = None

#         if 'date_range' in entities:
#             entities['date_range'] = self._parse_relative_date(entities['date_range'])

#         return entities

#     def _parse_relative_date(self, term: str) -> dict:
#         """Parses relative dates into absolute ranges."""
#         today = datetime.now().date()
#         date_mapping = {
#             'today': (today, today),
#             'yesterday': (today - timedelta(days=1), today - timedelta(days=1)),
#             'last week': (today - timedelta(days=7), today),
#             'this month': (today.replace(day=1), today)
#         }

#         if isinstance(term, str) and term.lower() in date_mapping:
#             start, end = date_mapping[term.lower()]
#             return {'start': start.strftime('%Y-%m-%d'), 'end': end.strftime('%Y-%m-%d')}
        
#         return {'start': None, 'end': None}

#     def parse_query(self, query_text: str) -> dict:
#         """Generates structured JSON response based on the AI model's output."""
#         prompt = f"""
#         Convert this query to JSON:

#         Query: "{query_text}"

#         {self._get_schema_prompt()}

#         Output Format:
#         {{
#           "action": "driver_report|vehicle_report|violation_search|ticket_status",
#           "entities": {{
#             "license_number": str|null,
#             "license_plate": str|null,
#             "violation_type": str|null,
#             "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}|null,
#             "ticket_status": "PENDING|PAID|FAILED"|null
#           }},
#           "confidence": "high|medium|low",
#           "sql_hint": "Suggested JOINs: drivers → violations → vehicles"
#         }}
#         """

#         try:
#             response = self.model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.1,
#                     top_p=0.9,
#                     response_mime_type="application/json"
#                 ),
#                 safety_settings=self.safety_settings
#             )

#             # print("Raw Gemini Response:", response)  # Debugging

#             # Extract JSON response safely
#             if response and response.candidates:
#                 json_response = response.candidates[0].content.parts[0].text.strip()
#             else:
#                 logger.error("Gemini AI did not return valid candidates.")
#                 return {"action": "error", "error_message": "No candidates received from AI", "confidence": "low"}

#             result = json.loads(json_response)

#             # Standardize entities to avoid `NoneType` errors
#             result['entities'] = self._standardize_entities(result.get('entities', {}))

#             return result

#         except json.JSONDecodeError as e:
#             logger.error(f"JSON Parsing Error: {str(e)}")
#             return {"action": "error", "error_message": "Invalid JSON format from AI", "confidence": "low"}

#         except Exception as e:
#             logger.error(f"Gemini API Error: {str(e)}")
#             return {"action": "error", "error_message": str(e), "confidence": "low"}





import json
import logging
import re
from django.conf import settings
import google.generativeai as genai
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class GeminiTrafficParser:
    def __init__(self):
        if not hasattr(settings, 'GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY missing in Django settings")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config={
                "temperature": 0.2,  # Slightly higher for flexibility
                "top_p": 0.95,
                "response_mime_type": "application/json"
            }
        )
        
        # Predefined mappings for standardization
        self.violation_types = {
            'speed': 'SPEEDING',
            'speeding': 'SPEEDING',
            'helmet': 'NO_HELMET',
            'light': 'RED_LIGHT_VIOLATION',
            'red light': 'RED_LIGHT_VIOLATION',
            'license': 'LICENSE_VIOLATION'
        }
        
        self.status_mapping = {
            'unpaid': 'PENDING',
            'paid': 'PAID',
            'failed': 'FAILED',
            'pending': 'PENDING'
        }

    def _get_schema_context(self) -> str:
        """Returns complete schema with relationships"""
        return """
        Database Schema (Table → Fields):
        
        ### Core Entities:
        1. Drivers (d):
           - id, license_number, first_name, last_name, date_of_birth, sex, 
           - phone_number, nationality, license_issue_date, blood_type
        
        2. Vehicles (ve):
           - id, license_plate, make, model, year, color, vin, registration_date
        
        3. Violations (v):
           - id, driver_id, vehicle_id, violation_type_id, issued_by_officer_id,
           - location, created_at, updated_at
        
        4. Tickets (t):
           - violation_id, status (PENDING/PAID/FAILED), reference_code, note,
           - issued_at, updated_at
        
        5. ViolationTypes (vt):
           - id, name (e.g., SPEEDING), description, fine_amount
        
        6. Officers (o):
           - user_id, badge_number
        
        7. Addresses (a):
           - id, driver_id, region, woreda, house_number, street, city, postal_code

        ### Key Relationships:
        - Driver 1→N Violations
        - Vehicle 1→N Violations 
        - Violation 1→1 Ticket
        - Violation N→1 ViolationType
        - Driver 1→N Addresses
        """

    def _build_dynamic_prompt(self, query: str) -> str:
        """Constructs a context-rich prompt for Gemini"""
        return f"""
        You are a SQL expert for a traffic management system. Convert this natural language query to an optimized MySQL query.

        ## User Query:
        "{query}"

        ## Database Context:
        {self._get_schema_context()}

        ## Output Requirements (JSON):
        {{
          "query": "SELECT ...",  // Complete MySQL query
          "entities": {{
            // Extracted entities from query
            "license_number": str|null,
            "license_plate": str|null,
            "violation_type": str|null,
            "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}|null,
            "ticket_status": str|null,
            "officer_badge": str|null,
            "location": str|null,
            "vehicle_make": str|null,
            "driver_attribute": str|null  // e.g., "blood_type=A+"
          }},
          "confidence": "high|medium|low",
          "query_type": "driver|vehicle|violation|ticket|officer|combined"
        }}

        ## Rules:
        1. Use these table aliases:
           - d = driver_api_driver
           - ve = driver_api_vehicle
           - v = driver_api_violation
           - t = driver_api_ticket
           - vt = driver_api_violationtype
           - o = driver_api_officer
        
        2. Always include:
           - LIMIT 1000 unless specified otherwise
           - Relevant WHERE clauses for exact matches
           - JOIN conditions using table aliases
        
        3. For date ranges:
           - Use v.created_at for violation dates
           - Support relative terms ("last month", "yesterday")
        
        4. Standardize values:
           - UPPERCASE for license numbers/plates
           - Predefined violation types
           - Mapped ticket statuses
        """

    def _standardize_entities(self, entities: Dict) -> Dict:
        """Normalizes all extracted entities"""
        # License/Plate standardization
        for field in ['license_number', 'license_plate']:
            if entities.get(field):
                entities[field] = re.sub(r'[^A-Z0-9]', '', str(entities[field])).upper()
        
        # Violation type mapping
        if entities.get('violation_type'):
            v_type = entities['violation_type'].lower()
            entities['violation_type'] = self.violation_types.get(v_type, v_type.upper())
        
        # Ticket status mapping
        if entities.get('ticket_status'):
            status = entities['ticket_status'].lower()
            entities['ticket_status'] = self.status_mapping.get(status, status.upper())
        
        # Date parsing
        if entities.get('date_range'):
            if isinstance(entities['date_range'], str):
                entities['date_range'] = self._parse_relative_date(entities['date_range'])
        
        return entities

    def _parse_relative_date(self, term: str) -> Dict:
        """Converts relative dates to absolute ranges"""
        today = datetime.now().date()
        term = term.lower()
        
        date_map = {
            'today': (today, today),
            'yesterday': (today - timedelta(days=1), today - timedelta(days=1)),
            'last week': (today - timedelta(days=7), today),
            'this month': (today.replace(day=1), today),
            'last month': (
                (today.replace(day=1) - timedelta(days=1)).replace(day=1),
                today.replace(day=1) - timedelta(days=1)
            ),
            'this year': (today.replace(month=1, day=1), today)
        }
        
        if term in date_map:
            start, end = date_map[term]
            return {
                'start': start.strftime('%Y-%m-%d'),
                'end': end.strftime('%Y-%m-%d')
            }
        return {'start': None, 'end': None}

    def parse_query(self, query_text: str) -> Dict:
        """Main method to convert natural language to structured query"""
        try:
            prompt = self._build_dynamic_prompt(query_text)
            response = self.model.generate_content(prompt)
            
            # Clean and parse response
            json_str = response.text.strip().replace('```json', '').replace('```', '')
            result = json.loads(json_str)
            
            # Validate and standardize
            # if not result.get('query', '').upper().startswith('SELECT'):
            #     raise ValueError("Generated query must be SELECT operation")
            
            result['entities'] = self._standardize_entities(result.get('entities', {}))
            
            return {
                "status": "success",
                "generated_sql": result['query'],
                "entities": result['entities'],
                "query_type": result.get('query_type', 'combined'),
                "confidence": result.get('confidence', 'medium')
            }
            
        except Exception as e:
            logger.error(f"Query parsing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "confidence": "low"
            }