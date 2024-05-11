import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime',region_name='ap-south-1')

prompt = "What is the capital of India?"

kwargs = {
 "modelId": "meta.llama3-8b-instruct-v1:0",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"prompt\":\"What is the capital city of Australia?\",\"max_gen_len\":512,\"temperature\":0.5,\"top_p\":0.9}"
}

response = bedrock_runtime.invoke_model(**kwargs)
print(json.loads(response['body'].read()))