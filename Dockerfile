# Create container from base TF image
FROM tensorflow/tensorflow:latest

# Change directory to /model/
WORKDIR /model/

# Copy python script from host to container
COPY makeMobileNet.py .

# Run the Python script
# CMD ["python3", "./makeMobileNet.py"]
# CMD ["makeMobileNet.py"]
# ENTRYPOINT ["python3"]
RUN python3 makeMobileNet.py
