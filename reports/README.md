# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [X] Instrument your API with a couple of system metrics (M28)
* [X] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [X] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 31

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:

s24084, s204150, s240154, s240076, s204084

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Answer:

We used the third-party frameworks TIMM, sci-kit learn in our project. We used functionality train_test_split from modelselection, metrics from sklearn to split effectively the data for training and testing, and implement some metrics in our project. We used TIMM to effectively load in rest-net model and to open the possibility of using other models as well, with ease and to finetune these for our use case. The TIMM framework made it very easy to implement large CNN models as they have much and these can often perform much better than home made models. We ended up choosing RestNet 18 as it's struck a good balance between performance and inference speed according to TIMM's model list. Naturally another model selection could be better in terms of maybe both accuracy and inference, although this would be further investigation of models which was not the goal of this project. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer:

We used a requirements.txt file, to manage our most necessary libraries and frameworks. At the start we initialized that with 
$ pip list 
$ pip freeze > requirements.txt 
from the conda environment we have created for the project. During the implementation of the project we made sure that each one of use was updating their requirements.txt file and merge them all together at main branch. We ended up having lots of lines that thew weren't required for the project anymore so at the end of the project we used 
$ deptry . in order to find and discard any libraries or frameworks we do not need to include.
In addition to this, we have enabled dependabot, which should help keep packages used up to date. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Answer:

From the cookiecutter template we have filled out the data, src and tests folder. We have removed the notebooks folder because we did not use any notebook files in our project and also the file src/animals/visualize.py because we did not implement any visual out of our model . Finally we have removed folder configs because we used another subfolder inside src to host the config files required. We have added a cloudbuild_files folder that contains the cloud files and a folder named profiling in order to keep all the files about profiling. The cookie cutter template helped kickstart the project and managed files consistently across all members of the group.


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We only set one check in Github Actions for ruff which seems to consistently fail even when the code has been checked, but at least notify us of potential issues in the code formating. We do understand how important these practices are, especially in bigger projects. Keeping code clean and consistent makes it easier to read and work with, and it helps avoid mistakes or confusion between team members. While we mostly agreed informally on how to keep things consistent, following proper coding standards would have made things more organized. Typing and documentation are also really important in larger projects because they help explain the code and prevent errors early on. Typing catches issues while writing the code, and documentation helps everyone understand how things work, both for current team members and anyone joining later.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Answer:

In total we have implemented nine tests. Three of them are intended for testing the data, three for our 
model and the other three for our API backend.

For the data part, we made sure that the datasets were initialized in the correct way for both the image and the target. 
We have also included tests that calculate the standard deviation and check that the number of images intended for 
training, testing and validation datasets are correct.

For the model part, we have checked that the output shape matches the expected one for a batch of images and the same 
but with a different input size. Finally, we have tested the model with invalid inputs (e.g., incorrect number of 
channels).

For the API part, we are checking if the backend can be contacted, if it can classify pictures and if it has connection to the cloud bucket.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Currently, 72 of the project's code has been covered. Even reaching coverage levels extremely near to 100% does not 
ensure that the software is error- or problem-free, even though a higher coverage percentage is preferable. This 
drawback results from the fact that code coverage quantifies the amount of written code that is executed, but it ignores 
potential use cases, edge situations, and unforeseen circumstances that might occur in practical application. 
Furthermore, code coverage alone may not be sufficient to identify some bug types, such as those brought on by external 
dependencies or integration problems. Using Github Actions, some tests are ignored, due to missing data. Of course, we could have made it, so data samples are downloaded temporarily, such that these tests could be automatically done as well. Fortunately, these can still be done locally.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

To work with version control in our code we have used different branches and pull requests. The tasks were distributed 
among the different team members for further development. For this, each member had his own branch to do the development 
and testing if needed. When one or more tasks were completed, a pull request was created to the main branch. After that, 
the rest of the team members were in charge of bringing the changes from the main branch to their own branch. The advantage of this workflow, is that we allow other members of the team to be up to date with the code, and peer review each other's changes.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We did use DVC to track any changes in the dataset, although we did not update the dataset during our implementation. However, there is a significant advantage to using data version control in a project. It allows teams to keep track of changes made to the dataset over time, ensuring that any modifications, additions, or deletions are recorded and can be easily reverted if needed. This is particularly useful in collaborative projects where multiple team members may work with the same data.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Answer:

Our continuous integration (CI) setup is organized into a single workflow file designed to ensure comprehensive testing 
across multiple platforms and Python versions. The workflow, named Unit Tests and Ruff Tests, is triggered on every push and pull 
request to the main branch.

We use a matrix strategy to test the code across three operating systems: Ubuntu, Windows, and macOS, with two Python 
versions: 3.11 and 3.12. This ensures compatibility and robustness of our code across diverse environments.

The workflow for the Unit Tests is broken into several steps:
- Checkout Code: The code is pulled using the actions/checkout action.
- Set Up Python: We configure Python using actions/setup-python, specifying the Python version from the matrix and 
enabling pip caching. This caching reduces redundant downloads and speeds up the workflow.
- Install Dependencies: Dependencies are installed from the requirements.txt file, followed by the installation of the 
project itself. This step also verifies that the dependencies are correctly resolved.
- Check If Coverage Is Available: Before running the tests, the workflow ensures that coverage and pytest are installed.
- Run Tests & Coverage: Tests are executed using pytest, with coverage used to measure code coverage. A summary report 
is then generated to monitor the extent of test coverage.

Here's the link to the workflow file: [CI File](https://github.com/iasonrap/mlops31/blob/main/.github/workflows/tests.yaml)

The Ruff Tests follow the same workflow except, here we just pip install ruff and do ruff check . and ruff format .
This workflow file can be found in the same folder at: [CI File](https://github.com/iasonrap/mlops31/blob/main/.github/workflows/codecheck.yaml)

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We used hydra to organize our config file which contains all the hyperparameters for our model and its optimizers, which makes it easy for anyone to replicate our results as they do not have to dig in our code to find this information. To run an experiment in our case we would do $ invoke train after changing the configuration file.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We set a seed at the start of each model train and we split the data with a seed as well, which should ensure that every run with the same configuration file should produce the same results. Outside of this, we have made use of docker images which are always a direct replication of our repository such a docker image is created whenever we change main branch of our repository. When we run an experiment, we train the model according to the configuration file and save the model weights locally, to wandb and to a cloud bucket, which can be used by the API.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Answer:
![WANDB](figures/wandb.png)
![WANDB2](figures/wandb2.png)
WANDB was used to keep track whenever there would be changed to the model, to see how it would impact our validation accuracy and therefore model versioning as well to keep an optimal model and tracking what has already been tried to no waste more time in it. WANDB fortunately has a great visualization of the different validation accuracies but also a great insight into the resources used for each of the models. We did not spend that long on making the visualizations pretty, but we have metrics regarding the model's train and validation data throughout the training epochs, to avoid issues such as overfitting. While we did not do it, performing a parameter sweep could have been beneficial to increase the model accuracy - we did consider trying smaller models as well to increase inference speed.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest l batch_size=64`. r=1e-3Link to docker file: <weblink>*
>
> Answer:

Docker was used as the preferred way to train our model and ensure anything that worked for someone would work for others. We uploaded docker images to the Artifact registry of Google Cloud Platfrom, which massively simplified the launch of the front and backed of our app. We have the train image build automatically, however the API needs to be build manually and the train image has to be used by vertex AI manually as well. An example of a dockerfile can be seen in [CI File](https://github.com/iasonrap/mlops31/dockerfiles/train.dockerfile). Pulling from the image registry is done by 
$ docker pull europe-west1-docker.pkg.dev/mlops31/animals-artifacts/frontend:latest
$ docker pull europe-west1-docker.pkg.dev/mlops31/animals-artifacts/animals_classification:latest

For example for running the frontend you can then use
$ docker run -e PORT=8080 -p 8080:8080 europe-west1-docker.pkg.dev/mlops31/animals-artifacts/frontend:latest

You can also pull the training file by
$ docker pull europe-west1-docker.pkg.dev/mlops31/animals-artifacts/animal_train_gpu:latest
$ docker run europe-west1-docker.pkg.dev/mlops31/animals-artifacts/animal_train_gpu:latest

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

The main way to debug, especially in code was to look at the error messages and narrow down the place the error took place. After that understanding what could have gone wrong, especially considering latest changes done by you or if a pull request was done, what that could have changed. Logging messages properly could have helped with this, and would also help other users of the repository, such that they only get the logs they ask for (using logging levels). Outside of prints statements, we naturally also used Python's debugger tool, with breakpoints to narrow down where issues arise, and how to fix them interactively.
 We did do profiling for train.py and there was nothing major to fix besides the training takes too long for minimum improvements, our results have been logged under profile_results.prof - An improvement could be to preprocess all data instead of processing it during training. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Answer:

We used 
-Bucket to store our data and track the changes with DVC. In addition to our data, we also use a cloud bucket, to gain responses from users of our app, which are used for testing using Evidently.
-Artifact Registry to store and manage our Docker container images. These are referenced and pulled whenever we want to retrain our model or redeply our Cloud Run application.
- Vertex AI was used shortly for attempting to train our model, however we could not seem to get access to a T4 GPU, and training on CPU took a long time. Fortunately, our model was not large enough to not be trained rather fast locally. Of course, this service is necessary otherwise, and it would make use of the training image in the artifact registry.
-Cloud Build to trigger a Docker image build on Main Branch push.
-Cloud Run was used to host the backend and frontend of our application which can be found on [Frontend link](https://frontend-739688782639.europe-west1.run.app/)
-Cloud Functions is used to generate a report using Evidently. The report is twofold, and is generated as a zip-file, a report of how the model is performing and a test report regarding the data quality. To generate such a report, you can use:
$ curl -X GET "https://europe-west1-mlops31.cloudfunctions.net/generate_performance_report" -o reports.zip
This was made into a Cloud Function as a server is not needed for a single function.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the Compute Engine primarily for our API in Google Cloud Run, as we had issues with training the model on a GPU using Vertex AI. For our Google Cloud Runs, we used instances with 1 Gigabyte of memory and a single CPU, which turns out to work decently, however it is quite slow, so updating the quota of our project and including a GPU would probably be beneficial. Outside of these, we have not interacted directly with the virtual machines, as we did not find it necessary for this project.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Bucket for data](figures/bucket.png)
[Bucket for user input](figures/bucket_2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Artifact Registry](figures/registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Build History](figures/build.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We did not manage to train in the cloud due to issues with accessibility to a GPU, however we did manage to get it to train on a CPU. This run was cancelled before completion though, as it barely managed a single epoch in a whole hour of training. It struck us too late to try a different type of GPU, which could have perhaps worked, but our model was not large enough for it to be infeasible to do locally. Due to the size of the model, it could maybe have been a good idea to not just save an image of train.py on each push to the main branch, but also to set up an automatic training of this image, to dynamically change the model weights. 

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did manage to write an API for our model, both a frontend and a backend. For this, we used FastAPI and Streamlit. The backend has two POST functions, one (/classify/) which takes an image and classifies it, using the model weights that are stored in a cloud bucket and another (/post_data/) which takes a DataPayload from the frontend and posts the results to a cloud bucket for further use by evidently. Playing around with Streamlit, we tried to make it interesting and fun to use, and cross fingers that the users won't abuse it, as no security features are put in place. The streamlit app can be found hosted on [Frontend link](https://frontend-739688782639.europe-west1.run.app/). The backend is quite uninteresting. 

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

The deployment of the backend was done by using FastAPI to create the application. Serving the model locally allowed us to use the /docs to test functionality and connection of the model, making sure that whatever was done locally would be reflected before deploying it to the cloud. The best way of accessing the backend is by using the frontend, which mitigates the need for using curl commands, which can get quite lengthy. The frontend sends a package with a file name to the backend which loads the file and classifies it, before enabling further functionality which can be used to evaluate the model. To invoke the backend service directly, a user would call:
$ curl -X POST -F "file@path/to/file.jpg" https://animal-classify-app-739688782639.europe-west1.run.app/classify/
One could also abuse the post_data/ function if providing the proper datapackage.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We did load testing using locust, representing a large number of users and seeing the effect on the relative response times and error rates. When running with 50 users, we saw that the response times linearly increased which means the responses are queeing due to the amount of time it takes to handle each request.
The failure rate here is about 50\%. So likely it is not optimal to have 50 users at the same time. We then tried reducing the load to see how it would perform, and with 20 users the response time stabilizes but is still slow, at about 70 seconds at the 95th percentile, with 40\% failures and an unimpressive rps of 0.4. And when testing with 10 users these figures were similar, with our response time going down slightly to 60 s. However with only 1 user, we reach a response time of 240 ms, which is far more reasonable. With a relative failure rate of 40. When testing this in browser we see that the instantiation takes a long time, however, once up and running it is quite fast, like the 200 ms we saw. We believe this is the cause of our runtime issues in locust, as it reinstantiates and is very long to instantiate each time.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Monitoring was implemented in a way, such that anyone can ping a Cloud Function and get performance reports from the deployed model. The curl command for this is:
$ curl -X GET "https://europe-west1-mlops31.cloudfunctions.net/generate_performance_report" -o reports.zip
And gives insight in how the model is currently performing in comparison to the test set which the model is evaluated on after training. In a working environment, such a report should largely be generated automatically, and it gives good insight in how the model is used (as long as it is used within its own scope AND users are being truthful). While this is focused on the data quality and the performance of the model, further metrics could be good to have such as response time / latency of the requests the model is getting. As we saw from the load testing, the response time is quite long - having an alert system could help keep the model latency down by giving it, for example, a GPU. While data drifting could also be good to monitor, we did not find a proper use case for it in this project.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Iason (s240154) spent $2.71 primarily from hosting the dataset in a google cloud bucket. Asbjørn (s204084) spent around $1 on hosting a smaller google cloud bucket and the cloud run services, and other group members spent around $0.2 we estimate. The total running cost is $3.96 all of which are covered by credits. The distribution of costs in regards to services is $0.81 for the artifact registry, $0.01 for cloud functions, $1.75 for storage, $1.40 for cloud run and $0 for vertex ai.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

The frontend as mentioned previously. This was purely to make it easier to access the model, and a bit for fun. We tried to increase the model performance by doing quantization, compiling and pruning, but we could not get it to work properly unfortunately. 

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

The starting point of the diagram is our local setup, where we have integrated the source of the project, which contains the model, the training of the model, the evaluation of it and the api which can be called. Whenever we commit code to the main branch, an image of the train.py is uploaded the the Cloud Artifact registry, to be used to retrain the model in addition to unittests being performed to ensure that no code breaks the repository. While the model is training we are continuously logging to Wandb, and once it is finished we upload the model weights to Wandb and a google cloud bucket.

The google cloud buckets are central to this project as the data to train the model is also hosted there. To download it, one need to just run the data.py file. In addition to the data, the buckets also perform a central role in storing the usage data of the deployed model, which can be used to create a performance report by a cloud function. The reference data to the evidently report, can be created by evaluating the locally stored model weights, which uploads the reference to a cloud bucket.

To update the different parts of our project, one has to do so manually, for example, training the model from the uploaded image needs to be performed manually to update the model weights. After having the model weights updated, the app which pulls the model weights on startup, needs to be reset before it makes use of these. Naturally it would be beneficial, especially as it is not a very large model we are working with, to automate the deployment of the app and the training of the model whenever something is updated (and has passed all the unittests). 

[this figure](figures/project_diagram.png)

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

There were quite some issues with the requirements.txt, lots of libraries being used or added that were not used and having issues with versioning. Many of these issues transferred over to the creation of Docker Images which took quite long to debug, as the docker images themselves take quite long to make. The issues with requirements were solved by making use of the package deptry (pip install deptry) which prunes packages that are not imported in the current project. Other issues included deploying deploying the model to cloud run, as the ports were not very intuitive to set, and there were slight differences in how to run things locally (on Windows) and the Linux OS, which made for small discrepancies in how to test things locally. There are still challenges in regards to accessing some of the cloud buckets, which were solved by simply making them public to the internet, though this is largely not desireable, as it can prove quite problematic if accessed by the wrong type of people. 

Creating the Unittests were also not straightforward, as they had to be configured correctly, such that they did not fail just because they didn't reflect the actual code they should be testing for. This was fixed by meticulous debugging. The codecheck.yaml seems to consistently fail, even though "ruff check ." and "ruff format ." fails to return anything before doing a pull request to main. Why this is, we don't know. Overall, the project took quite a long time, especially the creation and deployment of the api, which required a lot of debugging both locally and in the cloud. 

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
Student s240056 was in charge about buliding the docker image and therefore extended to make sure the requirements were always in line with the code. Also took the task to do some profiling in the code to ensure efficient code.
Student s240154 was in charge of data. Downloaded them and then storage to google Bucket, implement the data.py file. Also initialized the git repo and the Google Cloud project, and then managed to merge all the PRs and create some processes in Google Cloud Platform.
Student 204084 was in charge of creating train.py, instantiating the model in model.py and connecting everything to wandb. In addition to this they were responsible for the deployment of the model, both backend and frontend, and making sure it was up and running. Further, monitoring using evidently was also deployed by this student. Outside of this, this student was fixing many small bugs in regards to unittesting and created the workflow.yaml file.    
Student s204150 was in charge of api testing locust based load testing, as well as analysis of said testing. Made initial readme, helped out with other parts of the project, filling out this document, and setup wandb.
