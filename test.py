import mimetypes
from google.cloud import storage
import io


uri = 'gs://rkiles-test/nest/docs/Nest_Power_Connector_Installation_Guide.pdf'

# Split the URI by the last '/'
parts = uri.rsplit('/', 1)

filename = parts[-1]


print(filename)

# Detect the MIME type from the downloaded bytes
mime_type, encoding = mimetypes.guess_type(filename)
print(f"Detected MIME type: {mime_type}")



