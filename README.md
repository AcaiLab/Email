# Running the Email Experiment Locally with LM Studio

This repository contains scripts for running prompt-injection experiments against a local LLM using **LM Studio**. The experiments load prompts from the LLMail dataset and evaluate whether the model attempts unsafe tool calls.

---

## 1. Prerequisites

Before running the experiments, install the following:

* Python **3.10+**
* Git
* LM Studio

Download LM Studio from its official website and install it on your machine.

---

## 2. Install and Load a Local Model

1. Open **LM Studio**.
2. Navigate to the **Models** tab.
3. Download a compatible model (for example: `Phi-3-mini` or similar instruction models).
4. Once downloaded, load the model in the **Chat** or **Local Server** tab.

<img width="1919" height="951" alt="image" src="https://github.com/user-attachments/assets/d2cd318c-6d65-406b-95ac-09cbbf07e0c8" />

---

## 3. Start the LM Studio Local Server

LM Studio exposes a local OpenAI-compatible API.

1. Go to the **Local Server** tab.
2. Enable the server.
3. Confirm the endpoint is running at:

```
http://127.0.0.1:1234/v1
```

<img width="1917" height="1132" alt="image" src="https://github.com/user-attachments/assets/f1a77e99-d0f8-4dfe-b7a4-879819d75536" />


---

## 4. Set Up the Python Environment

Create a virtual environment:

```
python -m venv venv
```

Activate it:

**Windows**

```
venv\Scripts\activate
```

Install dependencies:

```
pip install datasets pandas tqdm openai
```

---

## 5. Run the Experiment

Once LM Studio is running and the environment is set up, run:

```
python run_level1_experiment.py
```

The script will:

1. Load prompts from the LLMail injection dataset (currently just top 20)
2. Send prompts to the local LLM through the LM Studio API
3. Validate whether the model attempted a tool call
4. Save results to:

```
level1_results.csv
```
<img width="1834" height="876" alt="image" src="https://github.com/user-attachments/assets/3e98cb21-fb91-4040-b8f2-280e97c690b2" />


---

## 6. Output

After the experiment finishes, results will be saved as a CSV file containing:

* Model output
* Tool-call detection flags
* Attack success indicators

Example output file:

```
level1_results.csv
```

<img width="1917" height="1112" alt="image" src="https://github.com/user-attachments/assets/3b9807f3-7e13-4d3e-a1a1-88ff4291d172" />


---

## Notes

* Ensure LM Studio is **running before starting the experiment**, otherwise the script will fail to connect.
* Different models may behave differently when exposed to prompt injection attacks.

---

## Summary

Workflow:

```
Download model in LM Studio
        ↓
Start local API server
        ↓
Activate Python venv
        ↓
Run experiment script
        ↓
Analyze results
```

This setup allows you to safely test prompt-injection attacks against **local LLMs without relying on external APIs**.
