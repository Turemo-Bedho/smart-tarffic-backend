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
import boto3
from botocore.exceptions import ClientError

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
            
            # Generate signed URL for the PDF (optional, if private access is needed)
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            pdf_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{pdf_path}"
            # Uncomment the following to use signed URLs instead of public URLs
            # pdf_url = s3_client.generate_presigned_url(
            #     'get_object',
            #     Params={
            #         'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
            #         'Key': pdf_path
            #     },
            #     ExpiresIn=3600  # URL valid for 1 hour
            # )
            
            return {
                "status": "success",
                "data": data,
                "pdf_url": pdf_url,
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
        """Generates professional PDF report and uploads to S3"""
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
        buffer.seek(0)
        
        # Upload to S3
        try:
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            
            # Generate unique filename for S3
            filename = f"reports/report_{uuid4().hex}.pdf"
            
            # Upload the file to S3 without ACL
            s3_client.upload_fileobj(
                buffer,
                settings.AWS_STORAGE_BUCKET_NAME,
                filename,
                ExtraArgs={
                    'ContentType': 'application/pdf'
                }
            )
            
            logger.info(f"Successfully uploaded PDF to S3: {filename}")
            return filename
            
        except ClientError as e:
            logger.error(f"Failed to upload PDF to S3: {str(e)}")
            raise Exception(f"Failed to upload PDF to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while uploading PDF to S3: {str(e)}")
            raise Exception(f"Unexpected error while uploading PDF to S3: {str(e)}")