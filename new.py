import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import plotly.express as px
from functools import total_ordering
from dataclasses import dataclass
import random

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AlgoSort | Professional Sorter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA STRUCTURES & HELPERS
# ==========================================

@dataclass
class SortMetrics:
    algorithm: str
    comparisons: int
    swaps: int
    time_taken: float

@total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __lt__(self, other):
        return self.age < other.age
    
    def __repr__(self):
        return f"{self.name}\n({self.age})"

class DataHandler:
    @staticmethod
    def generate_random(data_type, count=15):
        if data_type == "Integer":
            return [random.randint(1, 100) for _ in range(count)]
        elif data_type == "Float":
            return [round(random.uniform(0.1, 10.0), 2) for _ in range(count)]
        elif data_type == "String":
            words = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honey", "Ice", "Jack", "Kiwi", "Lemon"]
            return random.sample(words, min(count, len(words)))
        elif data_type == "Custom Object":
            names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi"]
            return [Person(n, random.randint(18, 60)) for n in random.sample(names, min(count, len(names)))]
        return []

    @staticmethod
    def parse_input(user_input, data_type):
        try:
            if not user_input.strip():
                return []
            
            parts = user_input.split(',')
            
            if data_type == "Integer":
                return [int(x.strip()) for x in parts]
            elif data_type == "Float":
                return [float(x.strip()) for x in parts]
            elif data_type == "String":
                return [x.strip() for x in parts]
            elif data_type == "Custom Object":
                objects = []
                for item in parts:
                    if ':' in item:
                        name, age = item.strip().split(':')
                        objects.append(Person(name.strip(), int(age.strip())))
                return objects
        except Exception:
            return None
        return []

# ==========================================
# 3. SORTING ALGORITHMS
# ==========================================
class SortEngine:
    def __init__(self):
        self.stats = {"comparisons": 0, "swaps": 0}
        self.frames = []

    def reset(self):
        self.stats = {"comparisons": 0, "swaps": 0}
        self.frames = []

    def _record(self, arr, highlights=None):
        # Deep copy needed for objects/lists
        self.frames.append((list(arr), highlights))

    # --- Wrapper to run any algo ---
    def run(self, algo_name, arr, ascending=True):
        self.reset()
        start_time = time.time()
        
        target_func = getattr(self, algo_name.lower().replace(" ", "_"))
        sorted_arr = target_func(arr, ascending)
        
        end_time = time.time()
        return SortMetrics(algo_name, self.stats["comparisons"], self.stats["swaps"], end_time - start_time), self.frames

    # --- Algorithms ---
    def bubble_sort(self, arr, ascending):
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                self.stats["comparisons"] += 1
                condition = (arr[j] > arr[j + 1]) if ascending else (arr[j] < arr[j + 1])
                
                if condition:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.stats["swaps"] += 1
                    self._record(arr, (j, j+1))
                else:
                    self._record(arr, (j, j+1)) # Record check even if no swap
        return arr

    def insertion_sort(self, arr, ascending):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0:
                self.stats["comparisons"] += 1
                condition = (arr[j] > key) if ascending else (arr[j] < key)
                if condition:
                    arr[j + 1] = arr[j]
                    j -= 1
                    self.stats["swaps"] += 1
                    self._record(arr, (j+1, j+2))
                else:
                    break
            arr[j + 1] = key
            self._record(arr, (j+1, j+1))
        return arr

    def selection_sort(self, arr, ascending):
        n = len(arr)
        for i in range(n):
            idx = i
            for j in range(i + 1, n):
                self.stats["comparisons"] += 1
                condition = (arr[j] < arr[idx]) if ascending else (arr[j] > arr[idx])
                if condition:
                    idx = j
                self._record(arr, (i, j)) # Visualization check
            
            if idx != i:
                arr[i], arr[idx] = arr[idx], arr[i]
                self.stats["swaps"] += 1
                self._record(arr, (i, idx))
        return arr

    def quick_sort(self, arr, ascending):
        def partition(low, high):
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                self.stats["comparisons"] += 1
                condition = (arr[j] <= pivot) if ascending else (arr[j] >= pivot)
                if condition:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self.stats["swaps"] += 1
                    self._record(arr, (i, j))
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.stats["swaps"] += 1
            self._record(arr, (i+1, high))
            return i + 1

        def _quick_sort_recursive(low, high):
            if low < high:
                pi = partition(low, high)
                _quick_sort_recursive(low, pi - 1)
                _quick_sort_recursive(pi + 1, high)

        _quick_sort_recursive(0, len(arr) - 1)
        return arr

    def merge_sort(self, arr, ascending):
        def _merge(start, mid, end):
            left = arr[start:mid+1]
            right = arr[mid+1:end+1]
            i = j = 0
            k = start
            
            while i < len(left) and j < len(right):
                self.stats["comparisons"] += 1
                condition = (left[i] <= right[j]) if ascending else (left[i] >= right[j])
                if condition:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                self.stats["swaps"] += 1 # Technically assignments
                k += 1
                self._record(arr, (k, k)) # Highlight current write
            
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
                self.stats["swaps"] += 1
                self._record(arr, (k, k))

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
                self.stats["swaps"] += 1
                self._record(arr, (k, k))

        def _sort(start, end):
            if start < end:
                mid = (start + end) // 2
                _sort(start, mid)
                _sort(mid + 1, end)
                _merge(start, mid, end)

        _sort(0, len(arr) - 1)
        return arr

    def heap_sort(self, arr, ascending):
        n = len(arr)
        
        def heapify(n, i):
            extreme = i
            l = 2 * i + 1
            r = 2 * i + 2

            if l < n:
                self.stats["comparisons"] += 1
                if (arr[l] > arr[extreme]) if ascending else (arr[l] < arr[extreme]):
                    extreme = l
            if r < n:
                self.stats["comparisons"] += 1
                if (arr[r] > arr[extreme]) if ascending else (arr[r] < arr[extreme]):
                    extreme = r
            
            if extreme != i:
                arr[i], arr[extreme] = arr[extreme], arr[i]
                self.stats["swaps"] += 1
                self._record(arr, (i, extreme))
                heapify(n, extreme)

        for i in range(n // 2 - 1, -1, -1):
            heapify(n, i)
        
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.stats["swaps"] += 1
            self._record(arr, (i, 0))
            heapify(i, 0)
        return arr

    def shell_sort(self, arr, ascending):
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap:
                    self.stats["comparisons"] += 1
                    cond = (arr[j - gap] > temp) if ascending else (arr[j - gap] < temp)
                    if cond:
                        arr[j] = arr[j - gap]
                        self.stats["swaps"] += 1
                        j -= gap
                        self._record(arr, (j, i))
                    else:
                        break
                arr[j] = temp
                self._record(arr, (j, i))
            gap //= 2
        return arr

# ==========================================
# 4. VISUALIZATION
# ==========================================

def render_plot(data, highlights, data_type, ax):
    ax.clear()
    n = len(data)
    
    # Determine heights and labels
    if data_type in ["Integer", "Float"]:
        heights = data
        labels = [str(x) for x in data]
    elif data_type == "Custom Object":
        heights = [obj.age for obj in data]
        labels = [str(obj) for obj in data]
    else: # String
        # Use index as height for visual sort, label is the string
        sorted_ref = sorted(list(set(data)))
        heights = [sorted_ref.index(x) + 1 for x in data]
        labels = data

    colors = ['#4F8BF9'] * n
    if highlights:
        idx1, idx2 = highlights
        if 0 <= idx1 < n: colors[idx1] = '#FF4B4B'
        if 0 <= idx2 < n: colors[idx2] = '#FF4B4B'

    ax.bar(range(n), heights, color=colors, alpha=0.9)
    
    # Formatting
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45 if n > 10 else 0, ha='right', fontsize=9)
    ax.set_yticks([]) # Hide y axis for cleaner look
    
    # Value labels on top of bars
    if n < 20:
        for i, v in enumerate(heights):
            ax.text(i, v + (max(heights)*0.01), str(v) if data_type != "Custom Object" else "", 
                   ha='center', va='bottom', fontweight='bold', fontsize=8)

    ax.set_title("Current State", fontsize=14, fontweight='bold', color="#333")
    return ax

# ==========================================
# 5. MAIN UI LOGIC
# ==========================================

def main():
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_type = st.selectbox("Data Type", ["Integer", "Float", "String", "Custom Object"])
        
        # Data Input Area
        st.subheader("Data Input")
        input_container = st.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Randomize"):
                st.session_state['data_input'] = DataHandler.generate_random(data_type)
        with col2:
            if st.button("🧹 Clear"):
                st.session_state['data_input'] = []

        # Default values if not in session state
        if 'data_input' not in st.session_state:
            st.session_state['data_input'] = DataHandler.generate_random(data_type, count=10)

        # Formatting current data for text area
        current_data = st.session_state['data_input']
        if data_type == "Custom Object":
            str_val = ", ".join([repr(x).replace('\n', ':').replace('(', '').replace(')', '') for x in current_data])
        else:
            str_val = ", ".join(map(str, current_data))

        user_input = st.text_area("Values (comma separated)", value=str_val, height=100)
        
        # Sort Settings
        st.divider()
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        ascending = sort_order == "Ascending"
        
        speed = st.select_slider("Animation Speed", options=["Slow", "Medium", "Fast", "Ultra"], value="Medium")
        speed_map = {"Slow": 0.3, "Medium": 0.1, "Fast": 0.01, "Ultra": 0.0001}

        st.divider()
        st.info("ℹ️ **Tip:** Select specific algorithms below to compare performance.")

    # --- Main Page ---
    st.markdown('<div class="main-header">📊 AlgoSort Visualizer</div>', unsafe_allow_html=True)

    # Parse Data
    data = DataHandler.parse_input(user_input, data_type)
    
    if not data or len(data) < 2:
        st.warning("⚠️ Please enter at least 2 valid data points to start sorting.")
        return

    # Tabs for modes
    tab1, tab2 = st.tabs(["🎥 Live Visualization", "📈 Benchmark Comparison"])

    engine = SortEngine()
    algorithms = ["Bubble Sort", "Insertion Sort", "Selection Sort", "Merge Sort", "Quick Sort", "Heap Sort", "Shell Sort"]

    # --- Tab 1: Visualization ---
    with tab1:
        col_algo, col_action = st.columns([3, 1])
        with col_algo:
            selected_algo = st.selectbox("Select Algorithm to Visualize", algorithms)
        with col_action:
            st.write("") # Spacer
            start_btn = st.button("▶️ Start Animation", type="primary", use_container_width=True)

        plot_spot = st.empty()
        stats_spot = st.empty()

        if start_btn:
            # 1. Run Algorithm to get frames
            metrics, frames = engine.run(selected_algo, data.copy(), ascending)
            
            # 2. Setup Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 3. Animate Loop
            progress_bar = st.progress(0)
            
            for i, (arr_state, highlights) in enumerate(frames):
                render_plot(arr_state, highlights, data_type, ax)
                plot_spot.pyplot(fig)
                
                # Update stats live
                stats_spot.markdown(f"""
                <div style="display: flex; justify-content: center; gap: 20px;">
                    <span style="font-weight:bold; color:#4F8BF9">Step: {i+1}/{len(frames)}</span>
                    <span>🔄 Swaps: {engine.stats['swaps']}</span>
                    <span>🔎 Comparisons: {engine.stats['comparisons']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress((i + 1) / len(frames))
                time.sleep(speed_map[speed])
            
            plt.close(fig)
            st.success(f"✅ {selected_algo} Completed in {metrics.time_taken:.4f} seconds!")

        # Initial Static Preview
        elif not start_btn:
            fig, ax = plt.subplots(figsize=(10, 5))
            render_plot(data, None, data_type, ax)
            plot_spot.pyplot(fig)
            stats_spot.info("Ready to sort. Click Start Animation.")

    # --- Tab 2: Benchmark ---
    with tab2:
        st.subheader("🏆 Algorithm Tournament")
        st.caption("Running all algorithms on the current dataset to find the most efficient one.")
        
        if st.button("🚀 Run Benchmark"):
            results = []
            progress = st.progress(0)
            
            for idx, algo in enumerate(algorithms):
                # Run algo (no visual recording needed ideally, but re-using run for simplicity)
                metrics, _ = engine.run(algo, data.copy(), ascending)
                results.append(metrics)
                progress.progress((idx + 1) / len(algorithms))
            
            # Convert to DataFrame
            df = pd.DataFrame([vars(m) for m in results])
            
            # Determine Winner (Weighted score: Time usually most important, then comparisons)
            # Simple logic: Sort by Time
            df_sorted = df.sort_values(by="time_taken").reset_index(drop=True)
            winner = df_sorted.iloc[0]

            # --- Display Winner ---
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid #2ecc71;">
                <h3 style="margin:0; color:#27ae60">🏆 Winner: {winner['algorithm']}</h3>
                <p>Time: {winner['time_taken']:.6f}s | Swaps: {winner['swaps']} | Comps: {winner['comparisons']}</p>
            </div>
            <br>
            """, unsafe_allow_html=True)

            # --- Metrics Columns ---
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("⏱️ Speed (Time)")
                fig_time = px.bar(df_sorted, x='algorithm', y='time_taken', 
                                  color='time_taken', color_continuous_scale='RdYlGn_r',
                                  title="Execution Time (seconds)")
                fig_time.update_layout(xaxis_title="", showlegend=False)
                st.plotly_chart(fig_time, use_container_width=True)
            
            with c2:
                st.subheader("🔎 Efficiency (Comps)")
                fig_comp = px.bar(df.sort_values('comparisons'), x='algorithm', y='comparisons', 
                                  color='comparisons', color_continuous_scale='Blues',
                                  title="Total Comparisons")
                fig_comp.update_layout(xaxis_title="", showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True)

            with c3:
                st.subheader("🔄 Work (Swaps)")
                fig_swap = px.bar(df.sort_values('swaps'), x='algorithm', y='swaps', 
                                  color='swaps', color_continuous_scale='Oranges',
                                  title="Total Swaps")
                fig_swap.update_layout(xaxis_title="", showlegend=False)
                st.plotly_chart(fig_swap, use_container_width=True)
                
            # Detailed Table
            st.subheader("📋 Detailed Results")
            st.dataframe(df_sorted.style.highlight_min(axis=0, color='#d4edda').format({"time_taken": "{:.6f}"}), use_container_width=True)

if __name__ == "__main__":
    main()
