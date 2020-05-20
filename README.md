# Simple SageMaker and Boto Project Example

This repo contains one Jupyter notebook that lets you try to:

1. Use the SageMaker instances with script mode to train the TensorFlow deep learning model and to classify the Kuzushiji-MNIST or KMNIST images (`kmnist_cnn.ipynb`)

The dataset for the project was initially obtained from the [Kuzushiji Recognition](https://www.kaggle.com/c/kuzushiji-recognition/overview) competition at [Kaggle](https://www.kaggle.com/). Visit [ROIS-DS Center for Open Data in the Humanities](http://codh.rois.ac.jp/) for more detail.

# Prerequisites

`Python 3.7` or later with virtual environment is preferred.

Run the following command and install all the necessary packages including the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) and [AWS Python SDK](https://github.com/boto/boto3) a.k.a. Boto 3.

```shell
pip install -r requirements.txt
```

**Note**: As listed in `requirements.txt`, the project assumes that you install JupyterLab but you may opt to use Jupyter Notebook instead.

Make sure your AWS or SageMaker credentials are set properly and both `AWS ACCESS KEY ID` and `AWS SECRET ACCESS KEY` are provided. Also, before you run the notebooks, prepare `settings.yml` as follows:

```yaml
  user: YOUR_USERNAME
  sagemaker:
    account_id: YOUR_SAGEMAKER_ACCOUNT_ID
    role_name: YOUR_SAGEMAKER_ROLE_NAME
  s3:
    bucket:
      name: BUCKET_NAME
  instance:
    train:
      type: TRAIN_INSTANCE_TYPE
      count: TRAIN_INSTANCE_COUNT
    endpoint:
      name: ENDPOINT_NAME
      type: INITIAL_INSTANCE_TYPE
      count: INSTANCE_COUNT
      delete: True  # Set either 'True' or 'False'
```

# References

Official documentation for Boto 3 and SageMaker including the Python SDK and Jupyter Notebook examples is available at:

  - [Boto 3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
  - [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html)
  - [Amazon SagaMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
  - [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples)
