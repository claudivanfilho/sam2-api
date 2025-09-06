# SAM2 API with S3 Integration

## S3 Mask Upload Feature

The `/segment` endpoint now automatically uploads segmentation masks to AWS S3 and returns both the mask data and the S3 URL.

### Configuration

Set the following environment variables:

```bash
S3_BUCKET_NAME_STAGING=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your-aws-access-key-id  
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=us-east-1
```

### S3 Bucket Setup

1. Create an S3 bucket in your AWS account
2. Configure bucket permissions to allow public read access for uploaded masks:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::your-bucket-name/sam2-api-masks/*"
        }
    ]
}
```

3. Ensure your AWS credentials have `s3:PutObject` and `s3:PutObjectAcl` permissions

### API Response

The `/segment` endpoint now returns only the S3 URL:

```json
{
    "mask_url": "https://your-bucket.s3.region.amazonaws.com/masks/20240902_143022_abc12345.png"
}
```

### Features

- **URL-Only Response**: The API returns only the S3 URL of the uploaded mask (no raw mask data)
- **Automatic Upload**: Masks are automatically converted to PNG images and uploaded to S3
- **Unique Filenames**: Each mask gets a unique filename with timestamp and UUID
- **Public Access**: Uploaded masks are publicly accessible via the returned URL
- **Graceful Fallback**: If S3 is not configured, the API still works but `mask_url` will be `null`
- **Original Size**: Masks are resized to match the original image dimensions before upload

### File Organization

Masks are stored in the S3 bucket with the following structure:
```
your-bucket/
└── sam2-api-masks/
    ├── 20240902_143022_abc12345.png
    ├── 20240902_143025_def67890.png
    └── ...
```

### Error Handling

- If AWS credentials are missing, the API logs a warning and continues without S3 upload
- If S3 upload fails, the error is logged but the API still returns the mask data
- The `mask_url` field will be `null` if upload fails or S3 is not configured
