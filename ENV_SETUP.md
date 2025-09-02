# Environment Variables Setup

The SAM2 API uses environment variables for configuration, particularly for AWS S3 integration. The application automatically loads variables from a `.env` file using the `python-dotenv` library.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file with your AWS credentials:**
   ```bash
   # AWS S3 Configuration for mask uploads
   S3_BUCKET_NAME=my-sam2-bucket
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=wJa...
   AWS_REGION=us-east-1
   ```

3. **Start the API** - it will automatically load these variables on startup

## Configuration Methods

### Method 1: .env File (Recommended)

The API uses `python-dotenv` to automatically load variables from a `.env` file in the project root:

```bash
# Create .env file
cp .env.example .env

# Edit with your values
nano .env
```

**Advantages:**
- ✅ Automatic loading on startup
- ✅ Easy to manage and version (example file)
- ✅ Secure (excluded from git)
- ✅ Works in all environments

### Method 2: System Environment Variables

Set environment variables directly in your shell:

```bash
export S3_BUCKET_NAME="my-sam2-bucket"
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="wJa..."
export AWS_REGION="us-east-1"

# Run the API
python3 main.py
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET_NAME` | No | `sam2-api-masks` | S3 bucket name for storing segmentation masks |
| `AWS_ACCESS_KEY_ID` | Yes* | None | AWS access key ID with S3 permissions |
| `AWS_SECRET_ACCESS_KEY` | Yes* | None | AWS secret access key |
| `AWS_REGION` | No | `us-east-1` | AWS region where your S3 bucket is located |

**\*Required only for S3 functionality.** If not provided:
- The API will log: `⚠️ AWS credentials not found. S3 upload will be disabled.`
- The `/segment` endpoint will return `"mask_url": null`
- All other functionality works normally

## Variable Loading Process

The application loads environment variables in this priority order:

1. **System environment variables** (highest priority)
2. **Variables from `.env` file** (loaded by python-dotenv)
3. **Default values in code** (fallback)

### How it works in code:
```python
from dotenv import load_dotenv
import os

# Load .env file variables
load_dotenv()

# Get variables with fallbacks
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "sam2-api-masks")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")  # None if not set
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
```

## Testing Your Configuration

### Quick Test
```bash
cd /root/sam2-api
python3 -c "
import os
from dotenv import load_dotenv

print('🔍 Testing environment variable loading...')
print('Before loading .env:', os.getenv('S3_BUCKET_NAME', 'Not Set'))

load_dotenv()
print('After loading .env:', os.getenv('S3_BUCKET_NAME'))
print('AWS Region:', os.getenv('AWS_REGION'))
print('AWS Key ID:', os.getenv('AWS_ACCESS_KEY_ID')[:10] + '...' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set')
"
```

### Full API Test
```bash
# Start the API to see if S3 client initializes
python3 main.py
```

**Expected output if configured correctly:**
```
✅ S3 client initialized for bucket: your-bucket-name
Loading EVF-SAM2 model...
...
```

**Expected output if not configured:**
```
⚠️ AWS credentials not found. S3 upload will be disabled.
Loading EVF-SAM2 model...
...
```

### Test S3 Connectivity
```bash
# Test S3 access (requires boto3 and valid credentials)
python3 -c "
import boto3
import os
from dotenv import load_dotenv

load_dotenv()
try:
    s3 = boto3.client('s3', 
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    bucket = os.getenv('S3_BUCKET_NAME', 'sam2-api-masks')
    s3.head_bucket(Bucket=bucket)
    print(f'✅ Successfully connected to S3 bucket: {bucket}')
except Exception as e:
    print(f'❌ S3 connection failed: {e}')
"
```

## AWS S3 Setup Guide

### 1. Create S3 Bucket
```bash
# Using AWS CLI
aws s3 mb s3://your-sam2-bucket --region us-east-1

# Or use the AWS Console at https://s3.console.aws.amazon.com/
```

### 2. Configure Bucket Permissions
The bucket needs to allow public read access for the uploaded mask images:

**Bucket Policy** (replace `your-sam2-bucket` with your bucket name):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::your-sam2-bucket/sam2-api-masks/*"
        }
    ]
}
```

### 3. Create IAM User with S3 Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-sam2-bucket/*"
        }
    ]
}
```

### 4. Get Access Keys
1. Go to IAM → Users → Your User → Security Credentials
2. Create Access Key → Application running outside AWS
3. Copy the Access Key ID and Secret Access Key to your `.env` file

## Security Best Practices

### Production Deployment
- ✅ **Use IAM Roles** instead of access keys when running on AWS EC2/ECS/Lambda
- ✅ **Use AWS Secrets Manager** or Parameter Store for credential management
- ✅ **Rotate access keys** regularly
- ✅ **Use least privilege** - only grant necessary S3 permissions
- ✅ **Enable CloudTrail** for API access logging

### Development/Local
- ✅ **Never commit `.env` files** to version control (added to `.gitignore`)
- ✅ **Use separate AWS accounts** for dev/staging/production
- ✅ **Restrict bucket access** to specific IP ranges if possible
- ✅ **Monitor S3 usage** and costs

### .env File Security
```bash
# Set proper file permissions
chmod 600 .env

# Verify it's in .gitignore
grep ".env" .gitignore
```

## Troubleshooting

### Common Issues

**1. "AWS credentials not found"**
```bash
# Check if .env file exists and has correct format
cat .env

# Test loading manually
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('AWS_ACCESS_KEY_ID'))"
```

**2. "S3 upload failed: AccessDenied"**
- Check IAM user permissions
- Verify bucket policy allows uploads
- Ensure access keys are correct

**3. "S3 upload failed: NoSuchBucket"**
- Verify bucket name is correct
- Check if bucket exists in the specified region
- Ensure AWS_REGION matches bucket region

**4. Images not accessible via URL**
- Check bucket policy allows public read access
- Verify the mask was uploaded (check S3 console)
- Test URL directly in browser

### Debug Mode
Add debug logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your API to see detailed logs
```
