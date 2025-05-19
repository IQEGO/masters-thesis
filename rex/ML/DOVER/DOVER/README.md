You need to download DOVER (https://github.com/VQAssessment/DOVER) in here in order to get dover.onnx. It needs to be on the same level as downloaded (cloned) onnx_inference.py file. For mobile_converter.py to work, you only need to download the DOVER-Mobile.pth into /pretrained_weights with:
mkdir pretrained_weights 
cd pretrained_weights 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth