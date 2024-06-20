# Mini President Classifier
Encoder code to classify speeches between Obama, H. Bush, and W. Bush. Frontend is here: https://mini-president-classifier-fe.vercel.app/

### How To Run
`python main.py -mode [c | d | sc | sd | ecd]`
- `c` is for running Part 1 (ALREADY INCLUDES A SANITY CHECK AT END)
- `d` is for running Part 2 (ALREADY INCLUDES A SANITY CHECK AT END)
- `sc` is for running Part 1 with sanity checker ONLY
- `sd` is for running Part 2 with sanity checker ONLY
- `ecd` is for running Part 3 Extra Credit Decoder
 
 Obama is 0, W. Bush is 1, and H. Bush is 2

 ### How To Run Docker Container
 - Build Image: `docker build -t <image-name> ./`
 - Run Container: `docker run -d -p 5000:5000 --name mini-pres-classifier <image-name>:latest`

