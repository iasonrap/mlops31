# animals

In this project, we aim to develop an image classification model capable of categorizing images into ten distinct animal classes using the Animals-10 dataset. This dataset comprises approximately 28,000 images across categories such as dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, and elephant.

The project involves several key steps:

Data Preparation: We will download and organize the Animals-10 dataset, ensuring a clear directory structure for each animal category. Data augmentation techniques, such as random cropping, flipping, and rotation, may be applied to enhance model performance,
 and normalizing the image sizes.

Model Selection and Fine-Tuning: A pre-trained ViT model, specifically google/vit-base-patch16-224, may be selected. We will modify its classification head to accommodate the ten animal categories and fine-tune the model using our dataset. This process allows the model to adapt learned features to the specific characteristics of the Animals-10 dataset.

Evaluation and Optimization: The model's performance will be assessed using metrics like accuracy, precision, recall, and F1-score. Based on these evaluations, hyperparameters such as learning rate and batch size may be adjusted to optimize performance.

Deployment: Upon achieving satisfactory results, the trained model will be saved and an inference pipeline will be developed. This pipeline will preprocess input images and generate predictions, facilitating practical applications of the model.

By fine-tuning a pre-trained ViT model with the Animals-10 dataset, this project seeks to create an efficient and accurate image classifier for diverse animal species. This approach should meet all requiremenets, using frameworks from outside the course, training a model and testing it.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
