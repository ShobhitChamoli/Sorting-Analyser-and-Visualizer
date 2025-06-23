# 🔢 Optimal Sorting Visualizer

An interactive **Streamlit web app** that visualizes and compares **9 sorting algorithms** across different data types. It identifies the **optimal algorithm** based on **comparisons, swaps, and execution time**, making it both educational and performance-driven.

---

## 🚀 Features

- 📊 **Step-by-step animated visualizations** using `matplotlib`
- ⚖️ **Automatic best-algorithm selection** based on:
  - ✅ Minimum Comparisons
  - ✅ Minimum Swaps
  - ✅ Lowest Execution Time
- 🧮 **9 Sorting Algorithms Supported**:
  - Bubble Sort
  - Insertion Sort
  - Selection Sort
  - Merge Sort
  - Quick Sort
  - Heap Sort
  - Shell Sort
  - Counting Sort *(Integers only)*
  - Radix Sort *(Integers only)*
- 📦 **Data Types Supported**:
  - **Integer** – e.g., `42, 17, 8, 56`
  - **Float** – e.g., `3.14, 2.71, 0.5`
  - **String** – e.g., `apple, banana, cherry`
  - **Custom Object** – e.g., `Alice:25, Bob:30, Charlie:20`

---



## 📚 How It Works

1. **User Input**: Choose data type and enter comma-separated values
2. **Performance Evaluation**: Each algorithm runs 10 trials to compute average:
   - Swaps
   - Comparisons
   - Time
3. **Ranking**: Algorithms are ranked and the best one is selected
4. **Visualization**: Step-by-step animation of the sorting process is shown

---

## 🛠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Visualization**: `matplotlib.animation`
- **Backend Logic**: Python
- **Others**: `NumPy`, `tempfile`, `functools.total_ordering`

---

## 📂 Main Components

- `run_page()` – Main interface to run visualizer
- `visualize()` – Animates sorting steps frame-by-frame
- `SortStats` – Tracks number of swaps and comparisons
- `parse_input()` – Handles input parsing for all 4 data types
- Sorting Functions – Implemented for all 9 algorithms with animation tracking

---

## 📈 Best Algorithm Selection Criteria

- Algorithms are shortlisted if:
  - `comparisons ≤ average comparisons`
  - `swaps ≤ average swaps`
- Among shortlisted, the one with the **least time** is selected
- If no algorithm qualifies, one with **lowest execution time** is chosen

---

