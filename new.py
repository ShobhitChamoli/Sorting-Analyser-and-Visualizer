import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import tempfile
import numpy as np
from matplotlib import style
from functools import total_ordering

plt.style.use('bmh')
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

class SortStats:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0

# ---------- DATA TYPE HANDLERS ----------
def parse_input(user_input, data_type):
    if data_type == "Integer":
        return [int(x.strip()) for x in user_input.strip().split(',')]
    elif data_type == "Float":
        return [float(x.strip()) for x in user_input.strip().split(',')]
    elif data_type == "String":
        return [x.strip() for x in user_input.strip().split(',')]
    elif data_type == "Custom Object":
        # For custom objects, we expect input like: "name:age, name:age"
        # Example: "Alice:25, Bob:30, Charlie:20"
        objects = []
        for item in user_input.strip().split(','):
            name, age = item.strip().split(':')
            objects.append(Person(name.strip(), int(age.strip())))
        return objects
    return []

@total_ordering
class Person:
    """Example custom object for sorting demonstration"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.age == other.age
    
    def __lt__(self, other):
        return self.age < other.age
    
    def __repr__(self):
        return f"{self.name}:{self.age}"

# ---------- SORTING ALGORITHMS WITH ORDER CONTROL ----------
def bubble_sort(arr, stats, frames, ascending=True):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            stats.comparisons += 1
            if (arr[j] > arr[j + 1]) if ascending else (arr[j] < arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                stats.swaps += 1
                frames.append((arr.copy(), (j, j+1)))
            else:
                frames.append((arr.copy(), None))
    return arr

def insertion_sort(arr, stats, frames, ascending=True):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and ((arr[j] > key) if ascending else (arr[j] < key)):
            stats.comparisons += 1
            arr[j + 1] = arr[j]
            j -= 1
            stats.swaps += 1
            frames.append((arr.copy(), (j+1, j+2)))  
        arr[j + 1] = key
        frames.append((arr.copy(), None))
    return arr

def selection_sort(arr, stats, frames, ascending=True):
    n = len(arr)
    for i in range(n):
        sel_idx = i
        for j in range(i + 1, n):
            stats.comparisons += 1
            if (arr[j] < arr[sel_idx]) if ascending else (arr[j] > arr[sel_idx]):
                sel_idx = j
            frames.append((arr.copy(), None))
        if sel_idx != i:
            arr[i], arr[sel_idx] = arr[sel_idx], arr[i]
            stats.swaps += 1
            frames.append((arr.copy(), (i, sel_idx)))
        else:
            frames.append((arr.copy(), None))
    return arr

def merge_sort(arr, stats, frames, ascending=True):
    def merge_sort_recursive(sub_arr, start_idx):
        if len(sub_arr) > 1:
            mid = len(sub_arr) // 2
            L = sub_arr[:mid]
            R = sub_arr[mid:]

            merge_sort_recursive(L, start_idx)
            merge_sort_recursive(R, start_idx + mid)

            i = j = k = 0
            while i < len(L) and j < len(R):
                stats.comparisons += 1
                if (L[i] < R[j]) if ascending else (L[i] > R[j]):
                    sub_arr[k] = L[i]
                    i += 1
                else:
                    sub_arr[k] = R[j]
                    j += 1
                stats.swaps += 1
                arr[start_idx:start_idx+len(sub_arr)] = sub_arr
                frames.append((arr.copy(), None))
                k += 1

            while i < len(L):
                sub_arr[k] = L[i]
                i += 1
                k += 1
                stats.swaps += 1
                arr[start_idx:start_idx+len(sub_arr)] = sub_arr
                frames.append((arr.copy(), None))

            while j < len(R):
                sub_arr[k] = R[j]
                j += 1
                k += 1
                stats.swaps += 1
                arr[start_idx:start_idx+len(sub_arr)] = sub_arr
                frames.append((arr.copy(), None))

    merge_sort_recursive(arr, 0)
    return arr
def quick_sort(arr, stats, frames, ascending=True):
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            stats.comparisons += 1
            if (arr[j] <= pivot) if ascending else (arr[j] >= pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                stats.swaps += 1
                frames.append((arr.copy(), (i, j)))
            else:
                frames.append((arr.copy(), None))
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        stats.swaps += 1
        frames.append((arr.copy(), (i + 1, high)))
        return i + 1

    def quick_sort_recursive(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)

    quick_sort_recursive(0, len(arr) - 1)
    return arr

def heap_sort(arr, stats, frames, ascending=True):
    def heapify(n, i):
        extreme = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n:
            stats.comparisons += 1
            if (arr[l] > arr[extreme]) if ascending else (arr[l] < arr[extreme]):
                extreme = l
        if r < n:
            stats.comparisons += 1
            if (arr[r] > arr[extreme]) if ascending else (arr[r] < arr[extreme]):
                extreme = r
        if extreme != i:
            arr[i], arr[extreme] = arr[extreme], arr[i]
            stats.swaps += 1
            frames.append((arr.copy(), (i, extreme)))
            heapify(n, extreme)
        else:
            frames.append((arr.copy(), None))

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        stats.swaps += 1
        frames.append((arr.copy(), (i, 0)))
        heapify(i, 0)
    return arr if ascending else arr[::-1]

def counting_sort(arr, stats, frames, ascending=True):
    # Only works for integers
    if not all(isinstance(x, int) for x in arr):
        return arr
    
    max_val = max(arr)
    count = [0] * (max_val + 1)
    output = [0] * len(arr)

    for num in arr:
        count[num] += 1
        stats.comparisons += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1
        stats.swaps += 1
        frames.append((output.copy(), None))

    for i in range(len(arr)):
        arr[i] = output[i]

    return arr if ascending else arr[::-1]

def radix_sort(arr, stats, frames, ascending=True):
    # Only works for integers
    if not all(isinstance(x, int) for x in arr):
        return arr
        
    def counting_sort_exp(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
            stats.comparisons += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(n - 1, -1, -1):
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            stats.swaps += 1

        for i in range(n):
            arr[i] = output[i]
            frames.append((arr.copy(), None))

    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_exp(arr, exp)
        exp *= 10

    return arr if ascending else arr[::-1]

def shell_sort(arr, stats, frames, ascending=True):
    n = len(arr)
    
    # Generate Knuth's gap sequence: 1, 4, 13, ...
    gaps = []
    gap = 1
    while gap < n:
        gaps.insert(0, gap)
        gap = 3 * gap + 1

    # Start sorting with each gap
    for gap in gaps:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and ((arr[j - gap] > temp) if ascending else (arr[j - gap] < temp)):
                stats.comparisons += 1
                arr[j] = arr[j - gap]
                stats.swaps += 1
                frames.append((arr.copy(), (j, j - gap)))
                j -= gap
            # One final comparison if condition failed first time
            if j != i:
                stats.comparisons += 1
            arr[j] = temp
            frames.append((arr.copy(), (j, i)))
    return arr

# ---------------- Visualization ----------------
def visualize(frames, data_type):
    fig, ax = plt.subplots()
    n = len(frames[0][0])
    
    # Get max value for y-axis limit
    if data_type in ["Integer", "Float"]:
        max_val = max(frames[0][0])
        y_limit = max_val + (0.1 * max_val) if max_val != 0 else 10
    else:
        # For strings and objects, we'll use a fixed scale
        y_limit = len(frames[0][0]) + 2
    
    bar_rects = ax.bar(range(n), [i+1 for i in range(n)] if data_type in ["String", "Custom Object"] else frames[0][0], 
                      align="edge", color='skyblue')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, y_limit)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    value_texts = []
    for rect in bar_rects:
        height = rect.get_height()
        if data_type in ["Integer", "Float"]:
            txt = ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, 
                         f'{int(height) if data_type == "Integer" else height:.1f}', 
                         ha='center', va='bottom', fontweight='bold')
        else:
            idx = int(rect.get_x())
            txt = ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, 
                         str(frames[0][0][idx]), 
                         ha='center', va='bottom', fontweight='bold')
        value_texts.append(txt)

    def update_plot(data):
        arr, swap_idx = data
        for rect, val, txt in zip(bar_rects, range(n) if data_type in ["String", "Custom Object"] else arr, value_texts):
            if data_type in ["Integer", "Float"]:
                rect.set_height(val)
                txt.set_text(f'{int(val) if data_type == "Integer" else val:.1f}')
            else:
                idx = int(rect.get_x())
                txt.set_text(str(arr[idx]))
            
            rect.set_color('skyblue')
            txt.set_x(rect.get_x() + rect.get_width()/2)
            txt.set_y(rect.get_height() + 0.5)
            txt.set_color('black')

        if swap_idx is not None:
            i, j = swap_idx
            bar_rects[i].set_color('red')
            bar_rects[j].set_color('red')
            value_texts[i].set_color('red')
            value_texts[j].set_color('red')

        text.set_text(f"Step: {frames.index(data)}")

    ani = animation.FuncAnimation(fig, update_plot, frames=frames, interval=500, repeat=False)
    temp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    ani.save(temp_gif.name, writer='pillow')
    plt.close(fig)
    st.image(temp_gif.name)

# ---------------- Runner and UI ----------------
def run_algorithm(algorithm_func, arr, ascending):
    stats = SortStats()
    frames = [(arr.copy(), None)]
    start = time.time()
    algorithm_func(arr.copy(), stats, frames, ascending)
    end = time.time()
    return stats.comparisons, stats.swaps, end - start, frames

def user_guide():
    st.title("📘 User Guide")
    st.markdown("""
    This app demonstrates **9 sorting algorithms** with visual animation:

    ### Supported Data Types:
    - **Integers**: 42, 17, 8, 56, 23
    - **Floats**: 3.14, 2.71, 1.618, 0.5
    - **Strings**: apple, banana, cherry, date
    - **Custom Objects**: name:age pairs (e.g., Alice:25, Bob:30)

    ### Algorithms:
    - Bubble Sort
    - Insertion Sort
    - Selection Sort
    - Merge Sort
    - Quick Sort
    - Heap Sort
    - Counting Sort (integers only)
    - Radix Sort (integers only)
    - Shell Sort

    ### Selection Logic:
    - Best algorithm must have **minimum swaps AND comparisons**
    - Then it is selected based on the **lowest execution time**
    - If no such algorithm exists, then pick one with **lowest time only**

    ### How to Use:
    1. Go to "Run Sort Visualizer"
    2. Select data type and enter values
    3. Choose sort order and hit Run
    4. View performance + animation!
    """)

def run_page():
    st.title("⚙ Run Sort Visualizer")
    
    # Data type selection
    data_type = st.selectbox("Select Data Type", 
                           ["Integer", "Float", "String", "Custom Object"],
                           help="Choose the type of data you want to sort")
    
    # Input examples based on data type
    examples = {
        "Integer": "42, 17, 8, 56, 23, 91, 33, 5, 70, 12",
        "Float": "3.14, 2.71, 1.618, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0",
        "String": "apple, banana, cherry, date, elderberry, fig, grape, honeydew",
        "Custom Object": "Alice:25, Bob:30, Charlie:20, David:35, Eve:28"
    }
    
    user_input = st.text_input(f"Enter {data_type} values separated by commas:", examples[data_type])
    
    if data_type == "Custom Object":
        st.info("For custom objects, enter in format 'name:age, name:age'")
    
    sort_order = st.radio("Choose Sort Order", ["Ascending", "Descending"])
    num_trials = 10  # Number of trials for averaging

    if st.button("Run Sorting Visualization"):
        try:
            arr = parse_input(user_input, data_type)
            if not arr:
                st.error("Please enter valid values")
                return
                
            ascending = sort_order == "Ascending"

            algorithms = {
                "Bubble Sort": bubble_sort,
                "Insertion Sort": insertion_sort,
                "Selection Sort": selection_sort,
                "Merge Sort": merge_sort,
                "Quick Sort": quick_sort,
                "Heap Sort": heap_sort,
                "Shell Sort": shell_sort,
            }
            
            # Only add counting and radix sort for integers
            if data_type == "Integer":
                algorithms["Counting Sort"] = counting_sort
                algorithms["Radix Sort"] = radix_sort

            results = {}

            for name, func in algorithms.items():
                total_comps, total_swaps, total_time = 0, 0, 0
                all_frames = None

                for trial in range(num_trials):
                    comps, swaps, exec_time, frames = run_algorithm(func, arr.copy(), ascending)
                    total_comps += comps
                    total_swaps += swaps
                    total_time += exec_time
                    if trial == 0:
                        all_frames = frames

                results[name] = {
                    "comparisons": total_comps // num_trials,
                    "swaps": total_swaps // num_trials,
                    "time": total_time / num_trials,
                    "frames": all_frames
                }

            # Calculate averages
            avg_swaps = np.mean([v["swaps"] for v in results.values()])
            avg_comps = np.mean([v["comparisons"] for v in results.values()])

            # Filter candidates
            candidates = {
                name: data for name, data in results.items()
                if data["swaps"] <= avg_swaps and data["comparisons"] <= avg_comps
            }

            if candidates:
                best_algo = min(candidates.items(), key=lambda x: x[1]["time"])[0]
            else:
                best_algo = min(results.items(), key=lambda x: x[1]["time"])[0]

            st.success(f"✅ Best Algorithm: {best_algo}")

            # Rank algorithms
            ranked_algos = sorted(
                results.items(),
                key=lambda x: (
                    0 if x[1]["swaps"] <= avg_swaps and x[1]["comparisons"] <= avg_comps else 1,
                    x[1]["time"]
                )
            )

            # Prepare data for table with Rank
            ranked_table = []
            for rank, (name, data) in enumerate(ranked_algos, start=1):
                ranked_table.append({
                    "Rank": rank,
                    "Algorithm": name,
                    "Swaps": data["swaps"],
                    "Comparisons": data["comparisons"],
                    "Time (s)": f"{data['time']:.6f}"
                })

            st.markdown("### 🏆 Algorithm Rankings")
            st.table(ranked_table)

            st.markdown(f"### 🎞 Sorting Animation: {best_algo}")
            visualize(results[best_algo]["frames"], data_type)

        except ValueError as e:
            st.error(f"Invalid input format: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="Optimal Sorting Visualizer", layout="wide")
    page = st.sidebar.selectbox("Select Page", ["📘 User Guide", "⚙ Run Sort Visualizer"])
    if page == "📘 User Guide":
        user_guide()
    else:
        run_page()

if __name__ == "__main__":
    main()
