# Nash-Equilibrium-Based CPM Prediction for Optimal Bidding in First-Price Ad Auctions

## Run Steps

### 1. Python Version

This project uses **Python 3.12.1**.

### 2. Create and Activate Virtual Environment (macOS/Linux/Windows)

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
# Using pip
pip install -q -r requirements.txt

# Or using uv
uv sync
```

### 4. Run the Entire Pipeline

Make the script executable:

```bash
chmod +x run_all.sh
```

Run everything:

```bash
./run_all.sh
```

### 5. Build and Run with Docker

**Build Docker Image:**

```bash
docker build -t cis5200-adops .
```

**Run Docker Container:**

```bash
docker run --rm cis5200-adops
```