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
                "temperature": 0.2,
                "top_p": 0.95,
                "response_mime_type": "application/json"
            }
        )
        self.safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        
        # Predefined mappings for standardization
        self.violation_types = {
            'speed': 'SPEEDING',
            'speeding': 'SPEEDING',
            'helmet': 'NO_HELMET',
            'light': 'RED_LIGHT_VIOLATION',
            'red light': 'RED_LIGHT_VIOLATION',
            'license': 'LICENSE_VIOLATION',
            'no helmet': 'NO_HELMET',
            'red light violation': 'RED_LIGHT_VIOLATION',
            'license violation': 'LICENSE_VIOLATION'
        }
        
        self.status_mapping = {
            'unpaid': 'PENDING',
            'paid': 'PAID',
            'failed': 'FAILED',
            'pending': 'PENDING'
        }

    def _get_schema_context(self) -> str:
        """Returns complete schema with correct table names and relationships"""
        return """
        Database Schema (Table → Fields):

        ### Core Entities:
        1. driver_api_driver (d):
           - id, license_number, first_name, middle_name, last_name, date_of_birth, sex,
           - phone_number, nationality, license_issue_date, blood_type, profile_image,
           - created_at, updated_at

        2. driver_api_vehicle (ve):
           - id, license_plate, make, model, year, color, vin, registration_date,
           - created_at, updated_at

        3. driver_api_violation (v):
           - id, driver_id, vehicle_id, issued_by_officer_id, location,
           - created_at, updated_at

        4. driver_api_ticket (t):
           - violation_id (primary key), status (PENDING/PAID/FAILED), reference_code, note,
           - issued_at, updated_at

        5. driver_api_violationtype (vt):
           - id, name (e.g., SPEEDING), description, fine_amount, created_at, updated_at

        6. driver_api_officer (o):
           - user_id (primary key), badge_number

        7. driver_api_address (a):
           - id, driver_id, region, woreda, house_number, street, city, postal_code,
           - created_at, updated_at

        8. driver_api_violation_violation_type (v_vt):
           - id, violation_id, violationtype_id (many-to-many relationship table)

        ### Key Relationships:
        - Driver 1→N Violations
        - Vehicle 1→N Violations
        - Violation 1→1 Ticket
        - Violation N→M ViolationType (through v_vt table)
        - Driver 1→N Addresses
        - Violation N→1 Officer (issuer)
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
            "license_number": str|null,
            "license_plate": str|null,
            "violation_type": str|null,
            "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}|null,
            "ticket_status": str|null,
            "officer_badge": str|null,
            "location": str|null,
            "vehicle_make": str|null,
            "driver_attribute": str|null,
            "driver_name": str|null,
            "violation_id": int|null
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
           - a = driver_api_address
           - v_vt = driver_api_violation_violation_type

        2. Important Notes:
           - Violation to ViolationType is MANY-TO-MANY relationship (use v_vt join table)
           - Ticket has ONE-TO-ONE relationship with Violation
           - Officer is optional in Violation (can be NULL)

        3. Always include:
           - LIMIT 1000 unless user explicitly asks for a single record
           - Proper JOIN conditions using correct aliases
           - Handle NULL officer cases in violations

        4. For date ranges:
           - Use v.created_at for violation dates
           - Support natural date terms (e.g., "last week", "this year")

        5. Standardize values:
           - UPPERCASE for license numbers/plates
           - Predefined violation types
           - Mapped ticket statuses
           - Handle driver names (first_name, middle_name, last_name)
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
        
        # Handle driver name components
        if entities.get('driver_name'):
            name_parts = entities['driver_name'].split()
            if len(name_parts) >= 3:
                entities['first_name'] = name_parts[0]
                entities['middle_name'] = ' '.join(name_parts[1:-1])
                entities['last_name'] = name_parts[-1]
            elif len(name_parts) == 2:
                entities['first_name'] = name_parts[0]
                entities['last_name'] = name_parts[1]
            elif len(name_parts) == 1:
                entities['last_name'] = name_parts[0]
        
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
            'this year': (today.replace(month=1, day=1), today),
            'last year': (today.replace(year=today.year-1, month=1, day=1),
                         today.replace(year=today.year-1, month=12, day=31))
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
            if not result.get('query', '').upper().startswith('SELECT'):
                raise ValueError("Generated query must be SELECT operation")
            
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