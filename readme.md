# 🇮🇳 India Vaccine Supply Chain Optimization

### (MILP + NSGA-II | Operations Research + AI)

🔬 Research-level project for optimizing large-scale vaccine supply chains using **exact optimization and evolutionary algorithms**

---

## 📌 Problem Statement

Design an efficient vaccine distribution system across India:

* Multi-tier network: Supplier → Distribution Center → Hospital
* Cold-chain constraints (temperature-sensitive vaccines)
* Trade-offs between cost, waste, and environmental impact

👉 **Objective:** Simultaneously minimize:

* 💰 Total Cost
* 🧊 Vaccine Waste
* 🌍 CO₂ Emissions

---

## 📊 Scale of the Model

* 🏭 15 Suppliers
* 🏢 100 Distribution Centers
* 🏥 500 Hospitals
* 🌐 **615 Nodes**
* 🔗 **51,500 Routes (Arcs)**

---

## 🧠 Methods Used

### 🔹 MILP (Exact Optimization)

* Mathematical formulation solved using `scipy.milp` (HiGHS solver)
* Provides **provably optimal solutions** for small/medium instances
* Used as a **benchmark and lower bound**

---

### 🔹 NSGA-II (Multi-objective Optimization)

* Evolutionary algorithm for large-scale problems
* Optimizes:

  * Cost
  * Waste
  * Emissions
* Produces **Pareto-optimal solutions**

---

## 🔬 Research Contribution

This project combines **exact optimization (MILP)** and **metaheuristic methods (NSGA-II)**:

* MILP → Optimal solutions (limited scalability)
* LP relaxation → Lower bound for full-scale problem
* NSGA-II → Scalable near-optimal solutions

👉 Demonstrates how heuristic methods approximate optimal solutions in real-world logistics systems.

---

## ⚙️ How It Works

1. Generate supply chain network (suppliers, DCs, retailers)
2. Formulate MILP model for exact optimization
3. Apply NSGA-II for large-scale multi-objective optimization
4. Evaluate solutions based on:

   * Cost
   * Waste
   * Emissions
5. Generate Pareto front for decision analysis

---

## 📈 Results

* 💰 Best Cost: ~₹7 Cr
* 📉 Significant waste reduction
* 🌍 CO₂ emissions optimized (~18% improvement)
* 🏢 Optimal DCs opened: ~8

---

## 📊 Key Insights

* NSGA-II achieves near-optimal solutions compared to MILP bounds
* Trade-off exists between cost, waste, and emissions
* Increasing DCs reduces transport cost but increases fixed cost
* Pareto front enables better decision-making in logistics planning

---

## 📊 Visualizations

### 🔹 Convergence

![GA](graphs/ga_convergence.png)

### 🔹 Pareto Front (Cost vs Waste)

![Pareto](graphs/pareto_2d.png)

### 🔹 3D Pareto (Cost–Waste–Emissions)

![3D](graphs/pareto_3d.png)

### 🔹 Cost vs DCs

![Cost](graphs/cost_vs_dc.png)

### 🔹 Selected Distribution Centers

![DC](graphs/selected_dc.png)

---

## ⚙️ Key Features

* ✅ Distance-based decay modeling
* ✅ Dynamic refrigeration decision
* ✅ Strict capacity constraints
* ✅ Multi-objective optimization
* ✅ Large-scale real-world simulation
* ✅ Pareto front analysis

---

## 🧾 Project Structure

```
nsga2.py      → Multi-objective optimization  
milp.py       → Exact optimization model  
graphs/       → Visualization outputs  
data/         → Input datasets  
docs/         → Project report  
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python nsga2.py
```

---

## 🧪 Technologies Used

* Python
* NumPy, Pandas
* SciPy (MILP solver)
* Evolutionary Algorithms (NSGA-II)
* Matplotlib

---

## 🎯 Applications

* Vaccine distribution systems
* E-commerce logistics optimization
* Cold-chain supply management
* Sustainable transportation planning

---

## 👨‍💻 Author

**Safal Pathak**
Mechanical Engineering + AIML Minor
Optimization & AI Enthusiast

---

## ⭐ If you like this project, consider giving it a star!
