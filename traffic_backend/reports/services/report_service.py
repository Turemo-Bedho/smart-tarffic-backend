# import json
# import base64
# import logging
# from datetime import datetime
# from django.db import connection
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors

# logger = logging.getLogger(__name__)

# class TrafficReportGenerator:
#     def generate_report(self, query_text: str) -> dict:
#         """Main pipeline for report generation"""
#         from .gemini_service import GeminiTrafficParser
#         parser = GeminiTrafficParser()
#         parsed = parser.parse_query(query_text)

#         if parsed.get('confidence') == 'low':
#             logger.error(f"Low confidence parsing: {parsed.get('error_message')}")
#             return {"error": f"Parsing failed: {parsed.get('error_message')}"}

#         results = self._execute_sql(parsed)
#         pdf_bytes = self._create_pdf(query_text, parsed['entities'], results)

#         return {
#             'metadata': parsed,
#             'data': results,
#             'pdf_base64': base64.b64encode(pdf_bytes).decode('utf-8')
#         }

#     def _execute_sql(self, parsed: dict) -> list:
#         """Executes parameterized SQL queries for reports"""
#         base_query = """
#         SELECT d.license_number, d.first_name || ' ' || d.last_name as driver_name,
#                v.created_at as violation_date, vt.name as violation_type, vt.fine_amount,
#                ve.license_plate, ve.make || ' ' || ve.model as vehicle, t.status as ticket_status
#         FROM violations v
#         JOIN drivers d ON v.driver_id = d.id
#         JOIN vehicles ve ON v.vehicle_id = ve.id
#         JOIN violation_types vt ON v.violation_type_id = vt.id
#         LEFT JOIN tickets t ON v.id = t.violation_id
#         """
        
#         where_clauses = []
#         params = []
#         entities = parsed['entities']

#         if entities.get('license_number'):
#             where_clauses.append("d.license_number = %s")
#             params.append(entities['license_number'])

#         if entities.get('license_plate'):
#             where_clauses.append("ve.license_plate LIKE %s")
#             params.append(f"{entities['license_plate']}%")

#         if entities.get('violation_type'):
#             where_clauses.append("vt.name = %s")
#             params.append(entities['violation_type'])

#         if entities.get('ticket_status'):
#             where_clauses.append("t.status = %s")
#             params.append(entities['ticket_status'])

#         if entities.get('date_range'):
#             start, end = entities['date_range'].get('start'), entities['date_range'].get('end')
#             if start and end:
#                 where_clauses.append("v.created_at BETWEEN %s AND %s")
#                 params.extend([start, end])

#         if where_clauses:
#             base_query += " WHERE " + " AND ".join(where_clauses)

#         try:
#             with connection.cursor() as cursor:
#                 cursor.execute(base_query, params)
#                 columns = [col[0] for col in cursor.description]
#                 return [dict(zip(columns, row)) for row in cursor.fetchall()]
#         except Exception as e:
#             logger.exception(f"SQL Execution Error: {e}")
#             return []

#     def _create_pdf(self, query_text: str, filters: dict, results: list) -> bytes:
#         """Generates a traffic violation report as a PDF"""
#         buffer = BytesIO()
#         doc = SimpleDocTemplate(buffer, pagesize=letter)
#         styles = getSampleStyleSheet()
#         elements = []

#         # Report Header
#         elements.append(Paragraph("Traffic Violation Report", styles['Title']))
#         elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))
#         elements.append(Paragraph(f"Query: {query_text}", styles['Normal']))

#         # Filters Table
#         if filters:
#             filter_data = [['Filter', 'Value']]
#             for key, value in filters.items():
#                 filter_data.append([key, json.dumps(value) if isinstance(value, dict) else str(value)])

#             elements.append(Paragraph("Filters Applied:", styles['Heading2']))
#             elements.append(Table(filter_data, style=[
#                 ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#                 ('GRID', (0, 0), (-1, -1), 1, colors.black)
#             ]))

#         # Results Table
#         if results:
#             headers = list(results[0].keys())
#             table_data = [headers] + [[str(item[h]) for h in headers] for item in results]

#             elements.append(Paragraph("Results:", styles['Heading2']))
#             elements.append(Table(table_data, repeatRows=1, style=[
#                 ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
#                 ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
#                 ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#                 ('FONTSIZE', (0, 0), (-1, -1), 8),
#                 ('GRID', (0, 0), (-1, -1), 1, colors.black)
#             ]))
#         else:
#             elements.append(Paragraph("No results found", styles['Heading3']))

#         doc.build(elements)
#         buffer.seek(0)
#         return buffer.read()







# import os
# import json
# import logging
# from uuid import uuid4  # Generate unique filenames
# from datetime import datetime
# from django.conf import settings
# from django.db import connection
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors

# logger = logging.getLogger(__name__)

# class TrafficReportGenerator:
#     def generate_report(self, query_text: str) -> dict:
#         """Main pipeline for report generation"""
#         from .gemini_service import GeminiTrafficParser
#         parser = GeminiTrafficParser()
#         parsed = parser.parse_query(query_text)

#         # Use AI-generated SQL if confidence is HIGH; otherwise, fallback to base query
#         use_base_query = parsed.get('confidence') == 'low'
#         sql_query = parsed.get('sql_hint').replace("Drivers", "driver_api_driver") if not use_base_query else self._get_base_query(parsed)

#         # Execute SQL & Generate Report
#         results = self._execute_sql(sql_query)
#         pdf_filename = self._create_pdf(query_text, parsed['entities'], results)

#         # Generate Full Downloadable URL
#         full_pdf_url = f"http://127.0.0.1:8000{settings.MEDIA_URL}{pdf_filename}"

#         return {
#             'metadata': parsed,
#             'data': results,
#             'pdf_url': full_pdf_url  # Return downloadable PDF link
#         }

#     def _execute_sql(self, sql_query: str) -> list:
#         """Executes either AI-generated SQL or structured base query"""
#         if not sql_query:
#             logger.error("No valid SQL query found.")
#             return []
#         # ✅ Remove AI suggestions that break MySQL syntax
#         sql_query = sql_query.split("Suggested JOINs:")[0].strip()

#         try:
#             with connection.cursor() as cursor:
#                 cursor.execute(sql_query)
#                 columns = [col[0] for col in cursor.description]
#                 return [dict(zip(columns, row)) for row in cursor.fetchall()]
#         except Exception as e:
#             logger.exception(f"SQL Execution Error: {e}")
#             return []

#     def _get_base_query(self, parsed: dict) -> str:
#         """Generates a structured SQL query if AI confidence is too low"""
#         base_query = "SELECT license_number, first_name, last_name, phone_number FROM driver_api_driver"
#         filters = []
#         entities = parsed['entities']

#         if entities.get('license_number'):
#             filters.append(f"license_number = '{entities['license_number']}'")

#         if filters:
#             base_query += " WHERE " + " AND ".join(filters)

#         return base_query

#     def _create_pdf(self, query_text: str, filters: dict, results: list) -> str:
#         """Generates and saves a traffic report as a PDF"""
#         buffer = BytesIO()
#         doc = SimpleDocTemplate(buffer, pagesize=letter)
#         styles = getSampleStyleSheet()
#         elements = []

#         # Report Header
#         elements.append(Paragraph("Traffic Report", styles['Title']))
#         elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))
#         elements.append(Paragraph(f"Query: {query_text}", styles['Normal']))

#         # Filters Table
#         if filters:
#             filter_data = [['Filter', 'Value']]
#             for key, value in filters.items():
#                 filter_data.append([key, json.dumps(value) if isinstance(value, dict) else str(value)])

#             elements.append(Paragraph("Filters Applied:", styles['Heading2']))
#             elements.append(Table(filter_data, style=[
#                 ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#                 ('GRID', (0, 0), (-1, -1), 1, colors.black)
#             ]))

#         # Results Table
#         if results:
#             headers = list(results[0].keys())
#             table_data = [headers] + [[str(item[h]) for h in headers] for item in results]

#             elements.append(Paragraph("Results:", styles['Heading2']))
#             elements.append(Table(table_data, style=[
#                 ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
#                 ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
#                 ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#                 ('FONTSIZE', (0, 0), (-1, -1), 8),
#                 ('GRID', (0, 0), (-1, -1), 1, colors.black)
#             ]))
#         else:
#             elements.append(Paragraph("No results found", styles['Heading3']))

#         doc.build(elements)
#         buffer.seek(0)

#         # Save PDF to Media Folder
#         pdf_filename = f"traffic_report_{uuid4().hex}.pdf"
#         pdf_path = os.path.join(settings.MEDIA_ROOT, pdf_filename)

#         with open(pdf_path, "wb") as pdf_file:
#             pdf_file.write(buffer.read())

#         return pdf_filename





import os
import logging
from uuid import uuid4
from datetime import datetime
from django.conf import settings
from django.db import connection
from io import BytesIO
from reportlab.lib import colors, pagesizes
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)


class TrafficReportGenerator:
    def generate_report(self, query_text: str) -> Dict:
        """End-to-end report generation pipeline"""
        try:
            # Step 1: Parse query
            from .gemini_service import GeminiTrafficParser
            parser = GeminiTrafficParser()
            parsed = parser.parse_query(query_text)
            
            if parsed['status'] != 'success':
                return parsed
            
            # Step 2: Execute query
            data = self._execute_query(parsed['generated_sql'])
            
            # Step 3: Generate PDF
            pdf_path = self._create_pdf(
                query_text=query_text,
                sql_query=parsed['generated_sql'],
                results=data,
                query_type=parsed['query_type']
            )
            
            return {
                "status": "success",
                "data": data,
                "pdf_url": f"{settings.MEDIA_URL}{pdf_path}",
                "metadata": {
                    "query_type": parsed['query_type'],
                    "confidence": parsed['confidence'],
                    "entities": parsed['entities']
                }
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _execute_query(self, sql: str) -> List[Dict]:
        """Executes SQL with robust error handling"""
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                
                # Get column names even with JOINs
                columns = [col[0] for col in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    # Handle both tuple and dict-like rows
                    if isinstance(row, dict):
                        results.append(row)
                    else:
                        results.append(dict(zip(columns, row)))
                
                return results
                
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}\nQuery: {sql[:200]}...")
            raise ValueError(f"Database error: {str(e)}")

    def _create_pdf(self, query_text: str, sql_query: str, 
                   results: List[Dict], query_type: str) -> str:
        """Generates professional PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=pagesizes.letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        elements = []
        
        # Title based on query type
        title_map = {
            'driver': "Driver Report",
            'vehicle': "Vehicle Report",
            'violation': "Violation Report",
            'ticket': "Ticket Report",
            'officer': "Officer Report",
            'combined': "Traffic System Report"
        }
        title = title_map.get(query_type, "Traffic Report")
        
        # Header
        elements.append(Paragraph(title, styles['Title']))
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles['Italic']
        ))
        
        # Query Section
        elements.append(Paragraph("User Query:", styles['Heading2']))
        elements.append(Paragraph(query_text, styles['Normal']))
        
        # SQL Query (collapsible in UI)
        elements.append(Paragraph("Generated SQL:", styles['Heading3']))
        elements.append(Paragraph(sql_query, styles['Code']))
        
        # Results Table
        if results:
            headers = list(results[0].keys())
            table_data = [headers] + [[str(row.get(h, '')) for h in headers] for row in results]
            
            # Auto-scale column widths
            col_widths = [doc.width/len(headers)] * len(headers)
            
            elements.append(Paragraph(
                f"Results ({len(results)} records):", 
                styles['Heading2']
            ))
            
            table = Table(
                table_data,
                colWidths=col_widths,
                repeatRows=1,
                style=[
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
                    ('BOX', (0,0), (-1,-1), 1, colors.black)
                ]
            )
            elements.append(table)
        else:
            elements.append(Paragraph("No results found", styles['Heading3']))
        
        doc.build(elements)
        
        # Save to unique filename
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'reports'), exist_ok=True)
        filename = f"reports/report_{uuid4().hex}.pdf"
        full_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        with open(full_path, 'wb') as f:
            f.write(buffer.getvalue())
            
        return filename