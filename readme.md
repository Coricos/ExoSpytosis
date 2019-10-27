![LOGO](./assets/exospytosis-logo.png)

`Authors: Alex Nguyen, Brian Roach, Pengcheng Zhang, Meryll Dindin`

## Set up your environment:

```bash
git clone https://github.com/Coricos/ExoSpytosis.git
virtualenv -p python3 ExoSpytosis
cd ExoSpytosis
source bin/activate
pip install -r requirements.txt
```

## For Jupyter users, add the environment to your kernels:

```bash
pip install jupyter notebook ipython ipykernel
python -m ipykernel install --user --name=exospytosis
```

## What we aim at solving:

![LOGO](./assets/event-spotting.png)

Using fluorescent false neurotransmitters (FFNs), which are fluorescent analogs of natural ("true") neurotransmitters, we are able to label vesicles in live cells with enough signal to permit imaging at high frequencies (up to 800 Hz).

The methods provided here are designed to automatically detect exocytosis events in live cell recordings.

## Current methodology:

![LOGO](./assets/baseline-removal.png)

![LOGO](./assets/initialization.png)

![LOGO](./assets/background-removal.png)

![LOGO](./assets/general-events.png)

![LOGO](./assets/roi-example.png)