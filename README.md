# cvml-mnist
Various experimentations with the MNIST data set and ML models

## Setup Instructions

* Setup virutal-env and setup dependencies
        
        virtualenv -p python3 venv
        source venv/bin/activate
        pip install -r requirements.txt

## Running the code

* To run the code without neural nets
        
        python src/mnsit_simple.py

* To run the code with neural nets
        
        python src/mnsit_cnn.py

## Saving results

* The code saves results in form of graphs as PNGs. The files are saved by the following scheme:

    * `{metric}_{model_type}_{batch_size}_{epochs}_{layers}{extra_params}.png`
    * `layers` and `extra_params` do not affect the experiment and are for file naming purposes only.
    * `batch_size` and `epochs` affect both the experiment and the filenaming scheme.
    * The only metrics saved are 'Accuracy' and 'Loss'
    * The graphs are plotted metric v/s epochs for both training and test data.

* Please change the parameters accordingly. 