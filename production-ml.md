= TOC

# Writing production ready ML code

Or, how to get and keep your work in production, **without losing sleep
at night**!By *David Josephs*

# About me!

## Supervillain origin story

### Graduated from UVA with degrees in aerospace engineering and astronomy

#### I hate planes and standing out in the cold at 4 am.

### Desperate need for career change

## Becoming a ML guy

### Went back to school, got masters

#### Thesis: Mapping electrical signals in the arm to hand motions, for smart prosthetics

##### Published to *ML4H 2020*

### Collaboration with NASA Goddard on automated spectroscopy, published to ACM 

## Academia

### \~1 year at Stanford

#### Deep representation/contrastive learning of multimodal medical data

## Current role

### Built out company\'s entire ML ecosystem from near scratch

### Models on both live and semi-live data, scaled across 5000+ machines

#### Marketed as flagship feature of our product

#### Live anomaly detection and unsupervised event categorization

##### Described as \"black magic\" by clients

#### Live pattern identification

#### Optimization

#### Signal segmentation and prediction

#### Current project: Reinforcement learning based optimization

# Overview

## What is this talk about?

### Picking data

### How do you track and debug models and code at a large scale?

### How do you set yourself up to get code into production quickly?

#### You learn these things through painful, avoidable trial and error

## What is this talk not about?

### How to pick a model

### How to tune a model

#### You will figure these things out with time

# Data Discussion

## How do I pick relevant data?

### Talk to someone with domain expertise! *number 1 tip*

> There is likely someone at your company who understands the field your
> company works

### When in doubt, grab more data (within reason)

## How do I know if my data is any good

### Figure out size of data (*step 0*)

What do your matrices look like? What sort of matrices make sense? This
informs your approach

### Make plots (*step 1!*)

#### Just plotting the data tells you a lot:

-   Is the data just crap?

-   Does the data need to be transformed (sometimes)

#### Correlation 

-   Are there any relationships or do i just have noise?

#### Show these plots to your expert!

Discuss with them! Find out what weird patterns mean

#### After this, plan out the rest of your exploration, and iterate with expert

Flow rate example

# Classes

Classes and good OOP costs you time up front, but later on you will be
thanking yourself

# Classes example

This is about 30 lines of code, to get a little model, which is the
absolute fastest way to develop things, there is little writing and
little cognitive overhead (for now)

      from some_module import query_clean_data
      from datetime import datetime, timedelta
      from sklearn.ensemble import IsolationForest
      
      winsize = 20
      
      time_dt = datetime.utcnow()
      from_time_dt = time_dt - timedelta(days=24)
      to_time_dt = time_dt + timedelta(days=1)
      to_time = to_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
      from_time = from_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
      query_dict = {
        "client_id": 'abc',
        "machine_id": 'machine_4',
        "from_time": from_time,
        "to_time": to_time,
        "description": ["Casing Pressure", "Flowrate", "Tubing Pressure"],
        "raw" = True
      }
      data = query_clean_data(query_dict)
      
      cp = np.array(data['Casing Pressure'])
      tp = np.array(data['Tubing Pressure'])
      load = cp - tp
      smoothed_load = np.zeros_like(load)
      for i in range(len(load)):
          lhs = max(0, i - winsize // 2)
          rhs = max(i + winsize // 2 + 1, winsize - lhs + 1)
          if rhs >= len(load):
              lhs -= rhs - len(load)
              rhs = len(load) - 1
          smoothed_load[i] = np.nanmedian(load[lhs:rhs])
      
      labels = IsolationForest.fit_predict(smoothed_load[:, np.newaxis])
      

This is totally great! Now wouldnt be funny if your boss asked you to:

-   Deploy this at a very large scale (running thousands of times per
    hour)

-   Using very questionable data

-   Make 10 other features just like this, but with actually complex and
    interesting features?

    > Are you really going to want to copy this script, edit the parts
    > you want, and pray it works?

# Classes example

Lets break this up into classes instead! First, we will be using a few
functions a lot!Every application in our stack is going to need:

-   from time and to time

-   some way to communicate with the server or AWS

-   query data

    -   Checks for if the data is there!

    Lets knock out the first 2!

In a new file, called maybe helpers.py, we define our class, which is
just a container for functions

      from datetime import datetime, timedelta
      
      class StackBase(object):
          def __init__(self, lookback):
              self.lookback = lookback
              self.from_time, self.to_time, self.time = self.get_time()
          
          def get_time(self):
              time_dt = datetime.datetime.utcnow()
              from_time_dt = time_dt - datetime.timedelta(days=self.lookback)
              to_time_dt = time_dt + datetime.timedelta(days=1)
              to_time = to_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
              from_time = from_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
              return from_time, to_time, time_dt.strftime("%Y-%m-%dT%H:%M:%S")
      
          @staticmethod
          def upload_to_dynamo(pred_table, payload):
              table = boto3.resource("dynamodb").Table(pred_table)
              table.put_item(TableName=pred_table, Item=payload)
              return
      

Next, lets rewrite our script!

      from helpers import StackBase
      from sklearn.ensemble import IsolationForest
      
      class LoadAD(StackBase):
          def __init__(self, lookback, client_id, machine_id, description):
              super().__init__(lookback)
              self.description = description
              self.query_dict = {
                  "client_id": client_id,
                  "machine_id": machine_id,
                  "from_time": self.from_time,
                  "to_time": self.to_time,
                  "description": self.description,
                  "raw": True
              }
      
          def load_data(self):
              self.data = query_clean_data(self.query_dict)
              return
      
          def run(self, winsize):
              load = np.array(self.data["Casing Pressure"]) - np.array(self.data["Tubing Pressure"])
              smoothed_load = np.zeros_like(load)
              for i in range(len(load)):
                  lhs = max(0, i - winsize // 2)
                  rhs = max(i + winsize // 2 + 1, winsize - lhs + 1)
                  if rhs >= len(load):
                      lhs -= rhs - len(load)
                      rhs = len(load) - 1
                  smoothed_load[i] = np.nanmedian(load[lhs:rhs])
              labels = IsolationForest.fit_predict(smoothed_load[:, np.newaxis])
              return labels
      
      
      AD = LoadAD(24, "221", "machine_3", ["Casing Pressure", "Tubing Pressure"])
      AD.load_data()
      result = AD.run()
      AD.upload_to_dynamo("secret_table", {"labels":result.tolist()})
      

# Classes example 

Since all clients will have a client id, machine_id, and some variables
to query, we can make our helpers even better.We can add a new subclass
to `helpers.py`, which works with any sort of data that comes out of the
`query_clean_data` function

      from datetime import datetime, timedelta
      from some_module import query_clean_data
      
      class CleanDataStack(StackBase): # note that we are subclassing not editing
          def __init__(self, lookback, client_id, machine_id, description, **kwargs):
              super().__init__(lookback)
              self.client_id = client_id
              self.machine_id = machine_id
              self.description = description
              self.query_dict = {
                  "client_id": client_id,
                  "machine_id": machine_id,
                  "from_time": self.from_time,
                  "to_time": self.to_time,
                  "description": self.description
              }
              self.query_dict = {**self.query_dict, **self.kwargs}
      
          def load_data(self):
              self.data = query_clean_data(self.query_dict)
              return
      
      

Now, we can easily use this class to get our data, for really any
application:

      from helpers import CleanDataStack
      from sklearn.ensemble import IsolationForest
      import numpy as np
      
      def handler(event, context):  # AWS event syntax
          stack = CleanDataStack(lookback=24,
                                  client_id=event['client_id'],
                                  machine_id=event['machine_id'],
                                  description=["Casing Pressure", "Tubing Pressure"],
                                  raw=True)
          stack.load_data()
          load = np.array(stack.data["Casing Pressure"]) - np.array(stack.data["Tubing Pressure"])
          winsize = 40
          smoothed_load = np.zeros_like(load)
          for i in range(len(load)):
              lhs = max(0, i - winsize // 2)
              rhs = max(i + winsize // 2 + 1, winsize - lhs + 1)
              if rhs >= len(load):
                  lhs -= rhs - len(load)
                  rhs = len(load) - 1
              smoothed_load[i] = np.nanmedian(load[lhs:rhs])
          labels = IsolationForest.fit_predict(smoothed_load[:, np.newaxis])
          stack.upload_to_dynamo("secret_table", {"labels":result.tolist()})
      

Our 30 lines of non reusable code has turned into about 15 lines of
readable, reusable code. With this, we can not only rapidly develop 1
model, but start building several, quickly!This framework for extending
classes through small subclasses can be applied as much as you want, any
group of related things you are going to use more than once belong in a
shared class!

# What is wrong with what we are doing?

## Are you sure this will work across thousands of data sources?

## How do you know its working?

### Are you just going to wait for a client to be angry at you?

# Tracking a model

## When you are working at a large scale, something bad will likely happen

> How do you quickly recover from issues, and debug?

## Keep track of the input data, training data, and model!

The simplest way is to use no external tools, however there are entire
tools built for this.This is how i manage to debug models quickly!

-   Use git!!

    -   When you deploy a model, save the git commit hash
        (e.g.`37c0f4a97b4249f83d414ff40b657d7704fe5ba5`)

    -   Add the hash to a file (e.g. a json file)

    -   Store training data in S3

        -   If training data was stored in a file, compress it, name it
            with the git hash, put in s3 bucket

        -   If training data was formed using many database queries,
            store the queries in a file, name with hash, put in bucket

    -   At runtime, when data goes in, store the query, or code used to
        make the query along with a dict in S3

        -   Strongly Recommended: save code for diagnostic plots in an
            s3 bucket, so you can diagnose the problem immediately

    -   Put all these s3 paths in a json and store in s3

        -   Put any unique identifiers (e.g. client_id, machine_id) in
            the filename, along with a timestamp

    -   If model is pickled, grab that pickle path too

    -   All this can be done with a simple CI/CD script

-   When something goes wrong:

    -   Use unique identifiers to grab json stored in s3

    -   Make a new branch on git (e.g.
        `git branch bug-2701; git checkout bug-2701`)

    -   Merge history into new branch:
        `git merge <Branch you started from>`

    -   Use json with s3 paths to reproduce the error, just write one
        python script

        -   Reproduce the erroneous prediction

        -   Use diagnostic plots to figure out why the prediction was
            the way it was

-   Resolving

    -   If you introduced this bug recently:

        -   `git reset --hard <hash-for-working-commit>`

    -   If this is a new bug:

        -   Fix it

    -   `git checkout <branch you started from>; git merge bug-2701`

        -   Retrain on whatever you need to retrain on

    -   Push your fix up!

# Tracking model performance

## You need to keep track of performance over time as well!

## Two things to look out for:

### Concept drift: input data is different from training data

#### This is when our hashed training and input data becomes useful

#### We can have a separate script that runs once a week to tell us how different training and production data are

#### At a certain threhsold:

##### You need to retrain your model

### Weird stuff: tracking prediction metrics over time

#### For example: ratio of positives vs all predictions

##### Script that notifies you every so often of this statistic 

##### Example: anomaly detection

##### If you are detecting anomalies on 50% of your data, something bad is probably happening

##### Figure out if the client is actually having a disaster or if something has changed in your data

# Tracking code performance!

## Use python logging!

### Your code likely interacts with other people\'s code, logs help you communicate with them

### You will likely have to debug non model related errors, e.g. 

-   database outages

-   your own mistakes

-   integrations with other components of company software

# Python logging demo

Lets update our base classes to have good logging. First lets update our
stack base:

      from datetime import datetime, timedelta
      import logging 
      
      class StackBase(object):
          def __init__(self, lookback):
              self.lookback = lookback
              self.from_time, self.to_time, self.time = self.get_time()
          
          def get_time(self):
              time_dt = datetime.datetime.utcnow()
              from_time_dt = time_dt - datetime.timedelta(days=self.lookback)
              to_time_dt = time_dt + datetime.timedelta(days=1)
              to_time = to_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
              from_time = from_time_dt.strftime("%Y-%m-%dT%H:%M:%S")
              ctime = time_dt.strftime("%Y-%m-%dT%H:%M:%S")
              logging.debug("[TIME] from: {}, to: {}, current: {}".format(from_time, to_time, ctime)
              return from_time, to_time, ctime
      
          @staticmethod
          def upload_to_dynamo(pred_table, payload):
              logging.info("Writing to {} with keys: {}".format(pred_table, list(payload.keys)))
              table = boto3.resource("dynamodb").Table(pred_table)
              table.put_item(TableName=pred_table, Item=payload)
              logging.info("success!")
              return
      

In this case, we may want to know (but not often, only in the case of
major database errors) what timespan we are querying, and we definitely
want to keep track of what we are sending to a table which may be used
by our coworkers.

# Python logging demo

Lets update our next class as well. The first thing we want to know, for
our data engineer, is how long our queries are taking!

      import logging
      import time 
      from some_module import query_clean_data
      
      def track_time(fn):
          def wrapped(*args, **kwargs):
              ts = time.time()
              ret = fn(*args, **kwargs)
              te = time.time()
              logging.info("{} executed in {} seconds".format(fn.__name__, round(te - ts, 2)))
              return ret
          return wrapped
      
      
      
      class CleanDataStack(StackBase): # note that we are subclassing not editing
          def __init__(self, lookback, client_id, machine_id, description, **kwargs):
              super().__init__(lookback)
              self.client_id = client_id
              self.machine_id = machine_id
              self.description = description
              self.query_dict = {
                  "client_id": client_id,
                  "machine_id": machine_id,
                  "from_time": self.from_time,
                  "to_time": self.to_time,
                  "description": self.description
              }
              self.query_dict = {**self.query_dict, **self.kwargs}
      
          @track_time
          def load_data(self):
              logging.debug("Querying with parameters: {}".format(self.query_dict))
              self.data = query_clean_data(self.query_dict)
              return
      

# Python logging demo

We also probably want to know whether or not the query returned data,
and if the data is fresh! We can implement that as follows:

      class CleanDataStack(StackBase): # note that we are subclassing not editing
          def __init__(self, lookback, client_id, machine_id, description, **kwargs):
              super().__init__(lookback)
              self.has_data = False # we have no data right now
              self.client_id = client_id
              self.machine_id = machine_id
              self.description = description
              self.query_dict = {
                  "client_id": client_id,
                  "machine_id": machine_id,
                  "from_time": self.from_time,
                  "to_time": self.to_time,
                  "description": self.description
              }
              self.query_dict = {**self.query_dict, **self.kwargs}
      
          @track_time
          def load_data(self):
              logging.debug("Querying with parameters: {}".format(self.query_dict))
              self.data = query_clean_data(self.query_dict)
              for k in self.description:
                  if k in self.data.keys():
                      if len(self.data[k]) > 0:
                          self.has_data = True
                          max_time = max(self.data["time"])
                          logging.info("Current time: {}, latest data time: {}".format(self.time, max_time))
                      else:
                          logging.critical("Data missing key {}".format(k))
                  else:
                      logging.critical("No data returned for key {}".format(k))
              return
      

Now, your data engineer will be able to understand what problems you are
having, and where the cause is quickly, and they will like you better,
and you will know when a problem is your problem vs someone else\'s!

# Python logging demo

Now lets wrap this up in our handler. We do not need to add any code at
that highest level, we just need to handle the missing data:

      from helpers import CleanDataStack
      from sklearn.ensemble import IsolationForest
      import numpy as np
      import decimal
      import logging
      
      root = logging.getLogger()
      if root.handlers:
          for h in root.handlers:
              root.removeHandler(h)
      logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
      
      def handler(event, context):  # AWS event syntax
          stack = CleanDataStack(lookback=event['lookback'],
                                  client_id=event['client_id'],
                                  machine_id=event['machine_id'],
                                  description=["Casing Pressure", "Tubing Pressure"],
                                  raw=True)
          ret_dict = {"labels": [], "client_id": event['client_id'], "machine_id": event['machine_id']}
          stack.load_data()
          if not stack.has_data:
              stack.upload_to_dynamo("secret_table", ret_dict)
              return
          load = np.array(stack.data["Casing Pressure"]) - np.array(stack.data["Tubing Pressure"])
          winsize = 40
          smoothed_load = np.zeros_like(load)
          for i in range(len(load)):
              lhs = max(0, i - winsize // 2)
              rhs = max(i + winsize // 2 + 1, winsize - lhs + 1)
              if rhs >= len(load):
                  lhs -= rhs - len(load)
                  rhs = len(load) - 1
              smoothed_load[i] = np.nanmedian(load[lhs:rhs])
          labels = IsolationForest.fit_predict(smoothed_load[:, np.newaxis])
          ret_dict['labels'] = [decimal.Decimal(x) for x in labels.tolist()]
          stack.upload_to_dynamo("secret_table", ret_dict)
          return
      

# Pydantic

This is great, but there are a few things annoying about this!

## How do you communicate the return data structure to coworkers

## What if you need to also have a json?

## What if you need to have a lot of specific types, or a very big (wide) database write?

## Are you going to copy and paste this ret_dict junk for every single thing you make?

## What if we need to ensure casing when we jsonify things?

This is where pydantic, which validates inputs and outputs comes in and
makes our life easier

# Pydantic example 

Lets go ahead and define a new file, `structures.py`:

      from pydantic import BaseModel, validator
      import typing
      import logging
      import decimal
      
      class Event(BaseModel):
          client_id: str
          machine_id: str  
      
      class LookBackEvent(Event):
          lookback: int
      
        @validator("lookback")
        def lookback_lt_9(cls, v):
            if v > 90:
                logging.warning("Cannot lookback more than 90 days. Setting lookback to 90")
                v = 90
            return v
      
      class LoadADOutput(LookBackEvent):
          labels: typing.List[decimal.Decimal] = [] # optional argument
      

# Pydantic example

Now lets integrate it into our handler:

      from helpers import CleanDataStack
      from sklearn.ensemble import IsolationForest
      import numpy as np
      import decimal
      from structures import LoadADOutput, LookBackEvent
      import logging
      
      root = logging.getLogger()
      if root.handlers:
          for h in root.handlers:
              root.removeHandler(h)
      logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
      
      
      def handler(event, context):  # AWS event syntax
          handled_event = LookBackEvent(**event)
          stack = CleanDataStack(
              **handled_event.dict(), description=["Casing Pressure", "Tubing Pressure"], raw=True)
          stack.load_data()
      
          if not stack.has_data:
              output = LoadADOutput(**handled_event.dict())
              stack.upload_to_dynamo("secret_table", output.dict())
              return
      
          load = np.array(stack.data["Casing Pressure"]) - np.array(stack.data["Tubing Pressure"])
          winsize = 40
          smoothed_load = np.zeros_like(load)
          for i in range(len(load)):
              lhs = max(0, i - winsize//2)
              rhs = max(i + winsize//2 + 1, winsize - lhs + 1)
              if rhs >= len(load):
                  lhs -= rhs - len(load)
                  rhs = len(load) - 1
              smoothed_load[i] = np.nanmedian(load[lhs:rhs])
          labels = IsolationForest.fit_predict(smoothed_load[:, np.newaxis])
      
          output = LoadADOutput(
              **handled_event.dict(), labels=labels.flatten().tolist())  # decimal conversion already done
          stack.upload_to_dynamo("secret_table", output.dict())
      
          # if we want some jsons or something:
          output_json = output.json()
          # if our frontend engineer wants a json schema so they can write typescript:
          json_schema = output.schema_json(indent=4)
          return
      

# Conclusion

## Motivation for classes 

-   maintainablity

-   can build out many models and tools quickly

## Logging and model tracking

-   Improve your own debug time, as well as your coworkers

## Pydantic and input/output validation

-   reduce code complexity

-   make it easier for someone else to use your code

-   enforce standards between applications

-   make your life easier

-   make sure users don\'t input stupid stuff

-   easy way to handle uncontrollable errors
