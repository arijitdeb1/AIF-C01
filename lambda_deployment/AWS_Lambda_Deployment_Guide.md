# FastAPI to AWS Lambda Deployment Guide

## Overview
This guide walks you through deploying your FastAPI application to AWS Lambda using Lambda Layers for dependencies.

## Prerequisites
- AWS CLI configured with appropriate permissions
- Python 3.12
- Docker (for building Lambda-compatible packages)
- AWS account with Lambda and API Gateway access

## Step 1: Prepare FastAPI for Lambda

### 1.1 Install Required Dependencies
```bash
uv add mangum
```

### 1.2 Create Lambda Handler
Create `lambda_deployment/lambda_handler.py`:
```python
from mangum import Mangum
from main import app

handler = Mangum(app)
```

### 1.3 Update main.py
Remove the `if __name__ == "__main__":` block from your main.py since Lambda will handle the execution.

## Step 2: Create Lambda Layer

### 2.1 Create Layer Directory Structure
```bash
mkdir lambda-layer
cd lambda-layer
# Note: The python directory will be created automatically by the Docker command
```

### 2.2 Install Dependencies for Lambda Layer
```bash
# Using Docker with AWS Lambda Python image for better compatibility

# For bash/Linux/macOS:
docker run --rm -v "${PWD}/..:/var/task" \
  --platform linux/amd64 \
  --entrypoint "" \
  public.ecr.aws/lambda/python:3.12 \
  /bin/sh -c "pip install --target /var/task/lambda-layer/python -r /var/task/requirements.txt --platform manylinux2014_x86_64 --only-binary=:all: --upgrade"

# For PowerShell:
docker run --rm -v "${PWD}/..:/var/task" `
  --platform linux/amd64 `
  --entrypoint "" `
  public.ecr.aws/lambda/python:3.12 `
  /bin/sh -c "pip install --target /var/task/lambda-layer/python -r /var/task/requirements.txt --platform manylinux2014_x86_64 --only-binary=:all: --upgrade"
```

Alternative without Docker (may have compatibility issues):
```bash
pip install -r ../requirements.txt -t python/
```

### 2.3 Create Layer ZIP
```bash
# For bash/Linux/macOS:
zip -r langchain-layer.zip python/

# For PowerShell:
Compress-Archive -Path .\python\ -DestinationPath langchain-layer.zip
```

### 2.4 Upload Layer to AWS Console
1. Go to AWS Lambda Console → Layers
2. Click "Create layer"
3. Layer name: `langchain-dependencies`
4. Upload `langchain-layer.zip`
5. Compatible architectures: `x86_64`
6. Compatible runtimes: `python3.12`
7. Click "Create"

## Step 3: Create Lambda Function

### 3.1 Create Deployment Package
```bash
# For bash/Linux/macOS:
cd lambda_deployment
zip -r lambda-function.zip . -x __pycache__/\* requirements.txt lambda-layer/\* *.zip *.md

# For PowerShell:
cd lambda_deployment
$files = Get-ChildItem -Path . -Name -Include "*.py"
Compress-Archive -Path $files -DestinationPath lambda-function.zip -Force
```

### 3.2 Create Lambda Function via AWS Console
1. Go to AWS Lambda Console
2. Click "Create function"
3. Function name: `fastapi-langchain-agent`
4. Runtime: `Python 3.12`
5. Upload `lambda-function.zip`
6. Handler: `lambda_handler.lambda_handler`

### 3.3 Add Layer to Function
1. In your Lambda function → Configuration → Layers
2. Click "Add a layer"
3. Select "Custom layers"
4. Choose your `langchain-dependencies` layer
5. Click "Add"

### 3.4 Configure Environment Variables
Add these environment variables in Lambda Configuration → Environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- Any other environment variables from your `.env` file

### 3.5 Increase Timeout and Memory
- Timeout: 5 minutes (300 seconds)
- Memory: 1024 MB (adjust based on your needs)

### 3.6 Deploy Lambda Function
1. After making any configuration changes, click "Deploy"
2. Wait for "Last modified" timestamp to update
3. Alternatively, if you updated your code:
   - Re-create the zip file using commands from 3.1
   - Upload the new zip: Code → Upload from → .zip file
   - Select your `lambda-function.zip`
   - Click "Save" then "Deploy"

### 3.7 Test Lambda Function Directly
1. In Lambda Console → Test tab
2. Click "Create new test event"
3. Event name: `test-fastapi-request`
4. Use this sample event template:
```json
{
  "version": "2.0",
  "routeKey": "POST /run-agent/",
  "rawPath": "/run-agent/",
  "rawQueryString": "",
  "headers": {
    "content-type": "application/json",
    "user-id": "test-user"
  },
  "requestContext": {
    "http": {
      "method": "POST",
      "path": "/run-agent/",
      "protocol": "HTTP/1.1",
      "sourceIp": "127.0.0.1"
    }
  },
  "body": "{\"question\": \"Add 5 and 10, then multiply the result by 2.\"}",
  "isBase64Encoded": false
}
```
5. Click "Save" then "Test"
6. Check the response and CloudWatch logs for any errors
7. If successful, you should see a 200 response with your agent's answer

**Troubleshooting:**
- Check CloudWatch logs if the function fails
- Ensure the Layer is properly attached
- Verify environment variables are set

## Step 4: Set Up API Gateway

### 4.1 Create HTTP API
1. In AWS Console, search for API Gateway
2. Click Create API
3. Choose HTTP API → Build
4. Step 1 - Create and configure integrations:
  - Click Add integration
  - Integration type: Lambda
  - Lambda function: Select twin-api from the dropdown
  - API name: twin-api-gateway
  - Click Next

### 4.2 Configure Routes
1. Step 2 - Configure routes:
2. You'll see a default route already created. Click Add route to add more:

Existing route (update it):

  - Method: `ANY`
  - Resource path: `/{proxy+}`
  - Integration target: `fastapi-langchain-api` (should already be selected)

Add these additional routes (click Add route for each):

Route 1:

  - Method: `GET`
  - Resource path: `/health`
  - Integration target: `fastapi-langchain-api`

Route 2:

  - Method: `POST`
  - Resource path: `/run-agent`
  - Integration target: `fastapi-langchain-api`

Route 3 (for CORS):

  - Method: `OPTIONS`
  - Resource path: `/{proxy+}`
  - Integration target: `fastapi-langchain-api`
3. Click Next

### 4.3 Configure Stages
1. Step 3 - Configure stages:
  - Stage name: `$default` (leave as is)
  - Auto-deploy: Leave enabled
2. Click Next

### 4.4 Review and Create
1. Step 4 - Review and create:
  - Review your configuration
  - You should see your Lambda integration and all routes listed
2. Click Create

### 4.5 Configure CORS
1. In your newly created API, go to CORS in the left menu
2. Click Configure
3. Settings:
  - Access-Control-Allow-Origin: Type * and click **Add** (important: you must click Add!)
  - Access-Control-Allow-Headers: Type * and click **Add** (don't just type - click Add!)
  - Access-Control-Allow-Methods: Type * and click **Add** (or add GET, POST, OPTIONS individually)
  - Access-Control-Max-Age: 300
4. Click Save

**Important**: For each field with multiple values (Origin, Headers, Methods), you must type the value and then click the Add button. The value won't be saved if you just type it without clicking Add!

## Step 5: Update Lambda Function Code

### 5.1 Final lambda_handler.py
```python
import json
from mangum import Mangum
from main import app

# Initialize Mangum handler
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        response = handler(event, context)
        return response
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
```

### 5.2 Update main.py for Lambda
```python
# Remove or comment out the uvicorn.run() part
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Step 6: Test Deployment

### 6.1 Get API Gateway URL
1. API Gateway Console → Your API → Stages → prod
2. Copy the Invoke URL

### 6.2 Test Health Endpoint
```bash
# For bash/Linux/macOS:
curl -X GET "https://your-api-id.execute-api.region.amazonaws.com/health"

# For PowerShell:
Invoke-RestMethod -Uri "https://your-api-id.execute-api.region.amazonaws.com/health" -Method GET
```

**Expected Response:**
```json
{"status":"healthy","service":"fastapi-langchain-agent"}
```

### 6.3 Test Agent Endpoint
```bash
# For bash/Linux/macOS:
curl -X POST "https://your-api-id.execute-api.region.amazonaws.com/prod/run-agent/" \
  -H "Content-Type: application/json" \
  -H "user-id: test-user" \
  -d '{"question": "Add 5 and 10, then multiply the result by 2."}'

# For PowerShell:
curl -X POST "https://your-api-id.execute-api.region.amazonaws.com/prod/run-agent/" -H "Content-Type: application/json" -H "user-id: test-user" -d "{""question"": ""Add 50 and 10, then multiply the result by 2.""}"
```

**Expected Response:**
```json
{"result":"The answer is 30. I added 5 and 10 to get 15, then multiplied by 2 to get 30."}
```

## Step 7: Troubleshooting

### 7.1 Common Issues
- **Cold Start Timeout**: Increase Lambda timeout
- **Memory Issues**: Increase Lambda memory allocation
- **Import Errors**: Ensure all dependencies are in the layer
- **Path Issues**: Use absolute imports in your code

### 7.2 CloudWatch Logs
- Check CloudWatch Logs for detailed error messages
- Lambda Console → Monitor → View CloudWatch logs

### 7.3 Layer Size Limits
- Layers have a 250MB limit (unzipped)
- If exceeded, consider splitting into multiple layers

## Step 8: Optimization Tips

### 8.1 Reduce Cold Start Time
- Use provisioned concurrency for consistent performance
- Minimize dependency imports
- Initialize shared resources outside the handler

### 8.2 Cost Optimization
- Use ARM-based Lambda (Graviton2) for better price/performance
- Set appropriate memory allocation
- Use reserved concurrency to control costs

## Step 9: CI/CD Pipeline (Optional)

### 9.1 GitHub Actions Example
```yaml
name: Deploy to Lambda
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Deploy to AWS
        run: |
          # Add your deployment commands here
```

## Conclusion
Your FastAPI application should now be deployed to AWS Lambda with proper dependency management through Lambda Layers. The API Gateway provides the HTTP endpoints for your application.

## Additional Resources
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Mangum Documentation](https://github.com/jordaneremieff/mangum)
- [API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)
