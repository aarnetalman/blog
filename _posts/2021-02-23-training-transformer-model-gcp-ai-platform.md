---
layout: post
title:  "Training PyTorch Transformers on GCP AI Platform"
---

Google Cloud Platform (GCP) is widely known for its great AI and machine learning capabilities and products. In fact there are tons of material available on how you can train and deploy TensorFlow models on GCP. However, GCP is not just for people using TensorFlow. It has good support for other frameworks as well. In this post I will show how to use another highly popular ML framework PyTorch on AI Platform Training. I will show how to fine-tune a state-of-the-art sequence classification model using PyTorch and the [`transformers`](HuggingFace Transformers) library. We will be using a pre-trained [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) as the transformer model for this task.

This post covers the following topics:
* How to structure your ML project for AI Platform Training
* Code for the model, the training routine and evaluation of the model
* How to launch and monitor your training job

You can find all the code on [Github](https://github.com/aarnetalman/transformers-sequence-classification-gcp).

## ML Project Structure

Let's start with the contents of our ML project. 

```
├── trainer/
│   ├── __init__.py
│   ├── experiment.py
│   ├── inputs.py
│   ├── model.py
│   └── task.py
├── scripts/
│   └── train-gcp.sh
├── config.yaml
└── setup.py
```
The `trainer` directory contains all the python files required to train the model. The contents of this directory will be packaged and submitted to AI platform. You can find more details and best practices on how to package your training application [here](https://cloud.google.com/ai-platform/training/docs/packaging-trainer). We will look at the contents of the individual files later in this post.

The `scripts` directory contains our training scripts that will configure the required environment variables and submit the job to AI Platform Training.

`config.yaml` contains configuration of the compute instance used for training the model. Finally, `setup.py`contains details about our python package and the required dependencies. AI Platform Training will use the details in this file to install any missing dependencies before starting the training job.

## PyTorch Code for Training the Model

Let's look at the contents of our python package. The first file, `__init__.py` is just an empty file. This needs to be in place and located in each subdirectory. The init files will be used by Python [Setuptools](https://setuptools.readthedocs.io/en/latest/) to identify directories with code to package. It is OK to leave this file empty. 

The rest of the files contain different parts of our PyTorch software. `task.py` is our main file and will be called by AI Platform Training. It retrieves the command line arguments for our training task and passes those to the `run` function in `experiment.py`.

```python
def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    parser = ArgumentParser(description='NLI with Transformers')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        type=int,
                        default=2)
    parser.add_argument('--log_every',
                        type=int,
                        default=50)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.00005)
    parser.add_argument('--fraction_of_train_data',
                        type=float,
                        default=1
                        )
    parser.add_argument('--seed',
                        type=int,
                        default=1234)
    parser.add_argument('--weight-decay',
                        default=0,
                        type=float)
    parser.add_argument('--job-dir',
                        help='GCS location to export models')
    parser.add_argument('--model-name',
                        help='The name of your saved model',
                        default='model.pth')

    return parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    experiment.run(args)


if __name__ == '__main__':
    main()

```

Before we look at the main training and evaluation routines, let's look at the `inputs.py` and `model.py` which define the datasets for the task and the transformer model respectively. First, the we use the [`datasets`](https://huggingface.co/docs/datasets/) library to retrieve our data for the experiment. We use the MultiNLI sequence classification dataset for this experiment. The `inputs.py` file contains code to retrieve, split and pre-process the data. The `NLIDataset` provides the PyTorch `Dataset` object for the training, development and test data for our task.

```python
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        #return len(self.labels)
        return len(self.encodings.input_ids)
```

The `load_data` function retrieves the data using the `datasets` library, splits the data into training, development and test sets, and then tokenises the input using `RobertaTokenizer` and creates PyTorch `DataLoader` objects for the different sets.

```python
def load_data(args):
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    nli_data = datasets.load_dataset('multi_nli')

    # For testing purposes get a slammer slice of the training data
    all_examples = len(nli_data['train']['label'])
    num_examples = int(round(all_examples * args.fraction_of_train_data))

    print("Training with {}/{} examples.".format(num_examples, all_examples))
    
    train_dataset = nli_data['train'][:num_examples]

    dev_dataset = nli_data['validation_matched']
    test_dataset = nli_data['validation_matched']

    train_labels = train_dataset['label']

    val_labels = dev_dataset['label']
    test_labels = test_dataset['label']

    train_encodings = tokenizer(train_dataset['premise'], train_dataset['hypothesis'], truncation=True, padding=True)
    val_encodings = tokenizer(dev_dataset['premise'], dev_dataset['hypothesis'], truncation=True, padding=True)
    test_encodings = tokenizer(test_dataset['premise'], test_dataset['hypothesis'], truncation=True, padding=True)

    train_dataset = NLIDataset(train_encodings, train_labels)
    val_dataset = NLIDataset(val_encodings, val_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader
```

The `save_model` function will save the trained model once it's been trained and uploads it to Google Cloud Storage.

```python
def save_model(args):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    bucket_name = args.job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix):].rstrip('/')

    datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')

    if bucket_path:
        model_path = '{}/{}/{}'.format(bucket_path, datetime_, args.model_name)
    else:
        model_path = '{}/{}'.format(datetime_, args.model_name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(args.model_name)
```

The `model.py` file contains code for the transformer model RoBERTa. The `create` function initialises the model and the optimiser.  

```python
# Specify the Transformer model
class RoBERTaModel(nn.Module):
    def __init__(self):
        """Defines the transformer model to be used.
        """
        super(RoBERTaModel, self).__init__()

        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    def forward(self, x, attention_mask, labels):
        return self.model(x, attention_mask=attention_mask, labels=labels)


def create(args, device):
    """
    Create the model

    Args:
      args: experiment parameters.
      device: device.
    """
    model = RoBERTaModel().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    return model, optimizer
```


The `experiment.py` file contains the main training and evaluation routines for our task. It contains the functions `train`, `evaluate` and `run`. The `train` function takes our training dataloader as an input and trains the model for one epoch in batches of the size defined in the command line arguments.

```python
def train(args, model, dataloader, optimizer, device):
    """Create the training loop for one epoch.

    Args:
      model: The transformer model that you are training, based on
      nn.Module
      dataloader: The training dataset
      optimizer: The selected optmizer to update parameters and gradients
      device: device
    """
    model.train()
    for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            if i == 0 or i % args.log_every == 0 or i+1 == len(dataloader):
                print("Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:<.4f}".format(
                    100. * (1+i) / len(dataloader), # Progress
                    i+1, len(dataloader), # Batch
                    loss.item())) # Loss
```

The `evaluate` function takes the development or test dataloader as an input and evaluates the prediction accuracy of our model. This will be called after each training epoch using the development dataloader and after the training has finished using the test dataloader.

```python
def evaluate(model, dataloader, device):
      """Create the evaluation loop.

    Args:
      model: The transformer model that you are training, based on
      nn.Module
      dataloader: The development or testing dataset
      device: device
    """
    print("\nStarting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for _, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = preds[1].argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch['labels'].cpu().numpy())

    print("Done evaluation")
    return np.concatenate(eval_labels), np.concatenate(eval_preds)
```

Finally, the `run` function calls the `run` and `evaluate` functions and saves the fine-tuned model to Google Cloud Storage once training has completed.

```python
def run(args):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.

    Args:
      args: experiment parameters.
    """
    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
      device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
      device = 'cpu'
    print('\n*************************')
    print('`cuda` available: {}'.format(cuda_availability))
    print('Current Device: {}'.format(device))
    print('*************************\n')

    torch.manual_seed(args.seed)

    # Open our dataset
    train_loader, eval_loader, test_loader = inputs.load_data(args)

    # Create the model, loss function, and optimizer
    bert_model, optimizer = model.create(args, device)

    # Train / Test the model
    for epoch in range(1, args.epochs + 1):
        train(args, bert_model, train_loader, optimizer, device)
        dev_labels, dev_preds = evaluate(bert_model, eval_loader, device)
        # Print validation accuracy
        dev_accuracy = (dev_labels == dev_preds).mean()
        print("\nDev accuracy after epoch {}: {}".format(epoch, dev_accuracy))

    # Evaluate the model
    print("Evaluate the model using the testing dataset")
    test_labels, test_preds = evaluate(bert_model, test_loader, device)
    # Print validation accuracy
    test_accuracy = (test_labels == test_preds).mean()
    print("\nTest accuracy after epoch {}: {}".format(args.epochs, test_accuracy))

    # Export the trained model
    torch.save(bert_model.state_dict(), args.model_name)

    # Save the model to GCS
    if args.job_dir:
        inputs.save_model(args)
```

## Launching and monitoring the training job

Once we have the python code for our training job, we need to prepare it for AI Platform Training. There are three important files required for this. First, `setup.py` contains information about the dependencies of our python package as well as metadata like name and version of the package.

```python
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'google-cloud-storage>=1.14.0',
    'transformers',
    'datasets',
    'numpy==1.18.5',
    'argparse',
    'tqdm==4.49.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Sequence Classification with Transformers on GCP AI Platform'
)
```

The `config.yaml` file contains information about the compute instance used for training the model. For this job we need use an NVIDIA V100 GPU as it provides improved training speed and larger GPU memory compared to the cheaper K80 GPUs. 

```yaml
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-standard-8
  masterConfig:
    acceleratorConfig:
      count: 1
      type: NVIDIA_TESLA_V100
```
Finally the `scripts` directory contains the `train-gcp.sh` script which includes the required environment variables as will as the gcloud command to submit the AI Platform Training job.

```bash
# BUCKET_NAME: unique bucket name
BUCKET_NAME=-name-of-your-gs-bucket

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=transformers_job_$(date +%Y%m%d_%H%M%S)

echo "Submitting AI Platform Training job: ${JOB_NAME}"

PACKAGE_PATH=./trainer # this can be a GCS location to a zipped and uploaded package

REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --config config.yaml \
    --job-dir ${JOB_DIR} \
    --module-name trainer.task \
    --package-path ${PACKAGE_PATH} \
    -- \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 2e-5

gcloud ai-platform jobs stream-logs ${JOB_NAME}
```

The list line of this script streams the logs directly to your command line. 

Alternatively you can head to GCP console and navigate to AI Platform jobs and select *View logs*.
![Logs](https://talman.io/images/ai-platform-logs.png)
You can also view the GPU utilisation and memory from the AI Platform job page.
![Monitoring GPU utilisation](https://talman.io/images/ai-platform-metrics.png)


## Conclusion

That concludes this post. You can find all the code on [Github](https://github.com/aarnetalman/transformers-sequence-classification-gcp). 

Hope you enjoyed this demo. Feel free to contact me if you have any questions.  
*   Twitter: [@AarneTalman](https://twitter.com/aarnetalman)
*   Website: [talman.io](https://talman.io)
