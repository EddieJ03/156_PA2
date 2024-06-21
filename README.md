# Mini President Classifier
Encoder code to classify speeches between Obama, H. Bush, and W. Bush. Frontend is here: https://mini-president-classifier-fe.vercel.app/
 
 Obama is 0, W. Bush is 1, and H. Bush is 2.

 ### How To Start & Run Server
 - Build Image: `docker build -t <image-name> ./`
 - Run Container: `docker run -d -p 5000:5000 --name mini-pres-classifier <image-name>:latest`

