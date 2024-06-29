# President Classifier
Encoder code to classify speeches between all presidents. Frontend is here: https://mini-president-classifier-fe.vercel.app/
 
HuggingFace Space: https://huggingface.co/spaces/edwjin/docker-classifier

Training done on a Google Colab (for that free TPU): https://colab.research.google.com/drive/15NDeBQ0AsHys_8Hbp1S81auwyJGGWw8M

Still working on ways to improve it!

 ### How To Start & Run Server
 - Build Image: `docker build -t <your-image-name> ./`
 - Run Container: `docker run -d -p 5000:5000 --name mini-pres-classifier <your-image-name>:latest`

