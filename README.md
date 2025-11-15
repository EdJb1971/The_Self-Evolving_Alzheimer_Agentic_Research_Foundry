# AlzNexus Agentic Autonomy Platform

![AlzNexus Logo](https://example.com/alznexus-logo.png) <!-- Placeholder for a project logo -->

## Project Overview

**AlzNexus** is a groundbreaking long-horizon agentic autonomy platform designed to revolutionize Alzheimer's Disease (AD) research. Leveraging a sophisticated master orchestrator and a swarm of specialized sub-agents, AlzNexus aims to conduct continuous, unsupervised research, generating end-to-end breakthroughs in biomarker discovery, clinical trial optimization, and drug repurposing. The platform is built with a strong commitment to privacy, ethics, and open-source principles, ensuring that all generated insights are verifiable, novel, and contribute to the public good.

Our primary objective is to seamlessly integrate with the AD Workbench, utilizing its secure APIs and federated queries to access and process vast amounts of AD-related data without compromising patient privacy. By automating the research lifecycle, from data scanning and hypothesis generation to validation and insight delivery, AlzNexus seeks to accelerate the pace of discovery and ultimately contribute to effective treatments and prevention strategies for Alzheimer's Disease.

**Key Stakeholders:**
*   Alzheimer's Disease Researchers
*   AD Workbench Platform Team
*   Open Source Community
*   Patients and Caregivers (indirect beneficiaries)
*   Funding Organizations/Prize Judges

## Features

AlzNexus is engineered with a comprehensive set of features to enable its autonomous research capabilities:

### Core Agentic Autonomy
*   **Master Orchestrator Agent:** The central intelligence responsible for setting high-level research goals, coordinating specialized sub-agents, managing multi-agent debates, and overseeing continuous, unsupervised research iterations. It initiates daily data scans and adapts strategies over long horizons (weeks/months).
*   **Specialized Sub-Agents:** A modular swarm of 8-12 domain-expert agents, including:
    *   **Biomarker Hunter:** Identifies novel early-stage signatures from imaging, omics, and clinical data.
    *   **Pathway Modeler:** Constructs and simulates novel disease progression models, identifying key intervention points.
    *   **Trial Optimizer:** Redesigns clinical trial parameters (e.g., inclusion criteria, endpoints) to simulate improved success probabilities.
    *   **Drug Screener:** Performs virtual screening of FDA-approved drugs, suggests de-novo targets, and utilizes molecular modeling.
    *   **Data Harmonizer:** Automatically and continuously aligns schemas across 50+ disparate studies within AD Workbench.
    *   **Hypothesis Validator:** Evaluates research hypotheses against supporting data, assessing statistical significance and biological plausibility.
    *   **Literature Bridger:** Scans scientific literature, extracts key information, and synthesizes connections between disparate research areas.
    *   **Collaboration Matchmaker:** Identifies complex research problems and suggests optimal multi-agent teams or external human experts for collaboration.
*   **Indefinite Autonomous Operation:** The platform is designed to wake up daily, scan for new data, set its own research goals, and iterate continuously without human intervention.
*   **Advanced Reasoning Loop:** Agents employ a full ReAct/Tool-use loop, reflection, and self-critique mechanisms to refine their approaches and learn from outcomes.
*   **Multi-Agent Collaboration and Debate:** Sub-agents can engage in structured debates to resolve conflicting information or evaluate alternative hypotheses, leading to more robust conclusions.
*   **Dynamic Agent Discovery and Registration Service:** Enables sub-agents to dynamically register their capabilities and API endpoints with the Master Orchestrator, enhancing platform flexibility and scalability.

### AD Workbench Integration & User Interaction
*   **Native AD Workbench Deployment:** Deployed as an official app/plugin within the AD Workbench marketplace for seamless access and installation.
*   **Secure Data Access:** Exclusively uses AD Workbench APIs and federated queries, ensuring no raw patient data is moved or exposed outside the secure environment.
*   **Secure Multi-User Mode:** Supports multiple researchers submitting queries and receiving responses securely.
*   **Proactive High-Value Insight Delivery:** Automatically pushes significant findings (e.g., new amyloid-tau-inflammatory axes) to relevant AD Workbench users.

### Ethical AI & Transparency
*   **Privacy-Preserving Reasoning:** All reasoning on sensitive data occurs via federated queries or secure enclaves.
*   **Full Audit Trail:** Maintains a comprehensive, immutable log of every decision, action, and reasoning step for transparency and accountability.
*   **Bias Detection and Correction:** A dedicated agent flags potential demographic imbalances or biases in data analysis and recommendations, proposing corrections.
*   **Centralized Audit Logging Service/Client:** Ensures consistent, secure, and maintainable audit trail entries across all microservices.

### Open Source & Public Good
*   **100% Open-Source:** The entire platform is released under an MIT/Apache license from day one.
*   **Model Weights/API Access:** Fine-tuned model weights are released or made accessible via API to ensure transparency and reusability.
*   **Zero Proprietary Lock-in:** Designed to avoid proprietary software components or vendor lock-in.

### Verifiable Insights
*   Demonstrates the generation of at least three publishable-quality, verifiably new insights live over a 7-14 day period.

## Architecture Summary

AlzNexus employs a robust microservices-oriented architecture designed for long-horizon agentic autonomy and seamless integration with the AD Workbench. The system is composed of several independent, yet interconnected, services orchestrated by a central Master Orchestrator.

### System Components:
*   **Master Orchestrator Service (COMP-001):** The core intelligence, responsible for high-level goal setting, sub-agent coordination, multi-agent debate resolution, and initiating autonomous research cycles. It interacts with the Agent Registry, Audit Trail, and dispatches tasks to sub-agents.
*   **Specialized Sub-Agent Services (COMP-002, COMP-013-016, etc.):** A collection of distinct microservices (e.g., Biomarker Hunter, Pathway Modeler, Trial Optimizer, Drug Screener, Data Harmonizer, Hypothesis Validator, Literature Bridger, Collaboration Matchmaker). Each agent possesses specialized domain expertise, implements ReAct/Tool-use loops, reflection, and self-critique, and registers itself with the Agent Registry.
*   **AD Workbench API Proxy Service (COMP-003):** A critical gateway for all interactions with the AD Workbench. It centralizes API calls, enforces secure federated queries, handles authentication/authorization, and ensures no raw patient data leaves the secure environment.
*   **Audit Trail Service (COMP-004):** Maintains a comprehensive, immutable log of every decision, action, reasoning step, and interaction across the platform, crucial for transparency and debugging.
*   **Insight Delivery Service (COMP-005):** Manages the proactive identification and delivery of high-value insights to relevant AD Workbench users, integrating with notification systems.
*   **Data Harmonization Service (COMP-006):** Continuously monitors and aligns schemas from 50+ disparate studies within AD Workbench, ensuring data consistency.
*   **LLM Service (COMP-007):** An abstraction layer for interacting with various Large Language Models (LLMs) via their APIs, handling prompt engineering and model selection.
*   **Database (PostgreSQL) (COMP-008):** The central persistent data store for agent states, research goals, audit logs, harmonized data schemas, and generated insights.
*   **Message Queue / Task Queue (RabbitMQ / Celery) (COMP-009):** Enables asynchronous communication between services and manages long-running background tasks, ensuring system responsiveness and scalability.
*   **Bias Detection Service (COMP-010):** A dedicated agent for continuously analyzing data inputs, agent reasoning, and generated outputs for potential biases, flagging issues and proposing corrections.
*   **Frontend Application (AD Workbench Plugin) (COMP-011):** A React-based user interface deployed as a native plugin within AD Workbench, providing secure multi-user query capabilities, displaying proactive insights, and allowing monitoring of agent activity and audit trails.
*   **Agent Registry Service (COMP-012):** A dedicated service for dynamic registration and discovery of specialized sub-agents, allowing the Master Orchestrator to query for available agents and their functionalities in real-time.

### Technology Stack:
*   **Frontend:** React, TypeScript, Tailwind CSS
*   **Backend:** FastAPI, Python
*   **Database:** PostgreSQL (with SQLAlchemy ORM)
*   **Asynchronous Tasks:** Celery with RabbitMQ as broker
*   **Caching/Rate Limiting:** Redis
*   **HTTP Client:** Axios (Frontend), Requests (Backend)

## How to Set Up & Run

This guide will walk you through setting up and running the AlzNexus platform locally using Docker Compose.

### Prerequisites
Before you begin, ensure you have the following installed:
*   **Docker Desktop:** Includes Docker Engine and Docker Compose. [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
*   **Python 3.9+:** For backend development and scripting.
*   **Node.js & npm (or yarn):** For frontend development.

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/alznexus.git # Replace with actual repository URL
cd alznexus
```

### 2. Environment Variables
Each service requires specific environment variables for configuration (database connections, API keys, Celery/Redis URLs). Create `.env` files in the root of each service directory (e.g., `src/backend/alznexus_orchestrator/.env`, `src/backend/alznexus_adworkbench_proxy/.env`, etc.).

**Example `.env` content for `src/backend/alznexus_orchestrator/.env`:**
```env
ORCHESTRATOR_DATABASE_URL="postgresql://user:password@db:5432/alznexus_db"
ORCHESTRATOR_CELERY_BROKER_URL="amqp://guest:guest@rabbitmq:5672//"
ORCHESTRATOR_CELERY_RESULT_BACKEND="redis://redis:6379/0"
ORCHESTRATOR_API_KEY="your_orchestrator_api_key"
ORCHESTRATOR_REDIS_URL="redis://redis:6379"
ADWORKBENCH_PROXY_URL="http://alznexus_adworkbench_proxy:8000"
ADWORKBENCH_API_KEY="your_adworkbench_api_key"
AUDIT_TRAIL_URL="http://alznexus_audit_trail:8000"
AUDIT_API_KEY="your_audit_api_key"
AGENT_SERVICE_BASE_URL="http://alznexus_biomarker_hunter_agent:8000" # Example, will be dynamic with registry
AGENT_API_KEY="your_agent_api_key"
```

**Important:**
*   Replace `user:password` with strong credentials for production.
*   Ensure `your_orchestrator_api_key`, `your_adworkbench_api_key`, `your_audit_api_key`, `your_agent_api_key`, and `your_registry_api_key` are unique and strong secrets.
*   For sub-agents, `AGENT_EXTERNAL_API_ENDPOINT` should be set to the externally accessible URL of the agent (e.g., `http://localhost:8000` if running locally, or the Docker service name if communicating within the Docker network).
*   The `AGENT_REGISTRY_URL` and `AGENT_REGISTRY_API_KEY` are crucial for agent self-registration.

### 3. Build and Run Docker Containers
Navigate to the root directory of the project (where `docker-compose.yml` is located) and run:

```bash
docker-compose build
docker-compose up -d
```

This command will:
*   Build Docker images for all services (backend, frontend, database, message queue, redis).
*   Start all services in detached mode (`-d`).

Wait a few minutes for all services, especially the databases, to initialize.

### 4. Initialize Databases (Manual Step for Development)
For development, you might need to manually create tables for each service. In a production environment, you would use database migration tools like Alembic.

Access the shell of each backend service container and run the `create_tables.py` script (if provided, or manually execute `Base.metadata.create_all(bind=engine)` in a Python shell within each container).

Example for Orchestrator:
```bash
docker exec -it alznexus_orchestrator_1 bash
python -c "from src.backend.alznexus_orchestrator.database import Base, engine; Base.metadata.create_all(bind=engine)"
exit
```
Repeat for `alznexus_adworkbench_proxy`, `alznexus_audit_trail`, `alznexus_agent_registry`, and each sub-agent (e.g., `alznexus_biomarker_hunter_agent`).

### 5. Access the Platform
Once all services are up and databases are initialized:

*   **Frontend Application:** Access the AlzNexus UI in your browser at `http://localhost:5173` (or the port configured in `vite.config.ts`).
    *   You can submit new research queries (STORY-201).
    *   Monitor the status of orchestrator tasks (STORY-203).
    *   View the audit trail for specific entities (STORY-204).

*   **Backend API Endpoints (Swagger UI):**
    *   **AD Workbench API Proxy:** `http://localhost:8000/docs`
    *   **Master Orchestrator:** `http://localhost:8001/docs`
    *   **Biomarker Hunter Agent:** `http://localhost:8002/docs`
    *   **Audit Trail Service:** `http://localhost:8003/docs`
    *   **Agent Registry Service:** `http://localhost:8004/docs`
    *   *(Other agents will have similar URLs, adjust port as per `docker-compose.yml`)*

    You can interact with the APIs directly using tools like `curl` or Postman, or through the Swagger UI.

### 6. Basic Usage Example (via Frontend)
1.  Open `http://localhost:5173`.
2.  Navigate to "Submit Query".
3.  Enter a research query, e.g., "Identify novel early-stage biomarkers for Alzheimer's disease."
4.  Click "Submit Query". You should see a "Query Submitted Successfully!" message with a Goal ID.
5.  Navigate to "Task Status".
6.  Enter the Goal ID from the previous step and click "Get Status". You should see the orchestrator task status update as it progresses through `PENDING`, `IN_PROGRESS`, and `COMPLETED` (after a simulated delay).
7.  Navigate to "Audit Trail".
8.  Enter `ORCHESTRATOR` for Entity Type and the Goal ID for Entity ID, then click "Search Audit History". You will see a detailed log of the orchestrator's actions related to your query.

### 7. Stopping the Platform
To stop all running Docker containers and remove their networks and volumes:

```bash
docker-compose down -v
```

This will remove all data volumes, so be cautious if you have important data. To stop without removing volumes, use `docker-compose down`.

## Contributing

We welcome contributions to AlzNexus! Please refer to our `CONTRIBUTING.md` (to be created) for guidelines on how to get involved.

## License

This project is licensed under the MIT License and Apache License 2.0. See the `LICENSE` file for details.

## Contact

For questions or support, please open an issue on the GitHub repository or contact [your-email@example.com].
