# Prosit: PROcess SImulation Tool

A Python Package for building and discovering Rule-Aware Business Process Simulation from event log data.


## 🛠️ How to Use

<ol>
    <li>
        <strong>Clone this repository.</strong>
    </li>
    <li>
        <strong>Create environment:</strong>
        <pre><code>$ conda env create -f environment.yml</code></pre>
    </li>
</ol>


```python
import prosit.simulator as simulator
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# read input Event log and Petri net
log = xes_importer.apply('data/logs/purchasing.xes')
net, initial_marking, final_marking = pm4py.read_pnml("data/models/purchasing.pnml")

# initialize simulation parameters
parameters = simulator.SimulatorParameters(net, initial_marking, final_marking)
# discover from event log
parameters.discover_from_eventlog(log)

# initialize simulation engine
simulator_eng = simulator.SimulatorEngine(parameters)
# simulate event log
sim_log = simulator_eng.apply(n_sim_traces, start_ts_simulation=log[0][0]['start:timestamp'])
```

## 📊 Evaluation Results

- **3GD**: 3-Gram Distance  
- **CTD**: Cycle Time Distribution Distance 
- **CAR**: Case Arrival Rate Distance 

🟩 = Best result, 🟥 = Worst result


<head>
  <meta charset="UTF-8">
  <title>ProSiT Evaluation Table</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #1e1e1e;
      color: #e0e0e0;
      display: flex;
      justify-content: center;
      padding: 40px;
    }
    table {
      max-width: 900px;
      width: 100%;
      border-collapse: collapse;
      background-color: #2a2a2a;
      font-size: 13px;
    }
    th, td {
      padding: 8px 10px;
      text-align: center;
      border: 1px solid #444;
    }
    th {
      background-color: #333;
      font-size: 14px;
    }
    .best {
      background-color: #228B22;
      font-weight: bold;
      color: white;
    }
    .worst {
      background-color: #B22222;
      color: white;
    }
    .prosit-col {
      background-color: #1b2f3a;
    }
    .spacer-row td {
      border: none;
      height: 3px;
      background-color: transparent;
    }
  </style>
</head>
<body>
<table>
  <thead>
    <tr>
      <th>Case Study</th>
      <th>Metric</th>
      <th>Simod</th>
      <th>DSim</th>
      <th>RIMS</th>
      <th class="prosit-col">ProSiT</th>
    </tr>
  </thead>
  <tbody>
    <tr class="spacer-row"><td colspan="6"></td></tr>
    <!-- P2P -->
    <tr><td rowspan="3">P2P</td><td>3GD</td><td>0.577218</td><td class="worst">0.596506</td><td>0.593391</td><td class="best">0.351867</td></tr>
    <tr><td>CTD</td><td>453.621311</td><td class="worst">584.295082</td><td>578.331066</td><td class="best">432.121311</td></tr>
    <tr><td>CAR</td><td>731.434426</td><td>770.538441</td><td class="worst">773.016897</td><td class="best">669.173770</td></tr>
    <tr class="spacer-row"><td colspan="6"></td></tr>
    <!-- ACR -->
    <tr><td rowspan="3">ACR</td><td>3GD</td><td class="worst">0.534745</td><td>0.249688</td><td>0.231299</td><td class="best">0.229476</td></tr>
    <tr><td>CTD</td><td class="worst">716.051309</td><td>70.190285</td><td class="best">46.282942</td><td class="prosit-col">190.077487</td></tr>
    <tr><td>CAR</td><td>250.330890</td><td class="worst">236.814151</td><td>233.661219</td><td class="best">180.235602</td></tr>
    <tr class="spacer-row"><td colspan="6"></td></tr>
    <!-- CVS -->
    <tr><td rowspan="3">CVS</td><td>3GD</td><td class="worst">0.581966</td><td>0.341521</td><td>0.384304</td><td class="best">0.124729</td></tr>
    <tr><td>CTD</td><td class="worst">269.929800</td><td class="best">52.429900</td><td>99.247400</td><td class="prosit-col">59.904300</td></tr>
    <tr><td>CAR</td><td class="best">5.394300</td><td class="worst">20.366700</td><td>20.262100</td><td class="prosit-col">17.777000</td></tr>
    <tr class="spacer-row"><td colspan="6"></td></tr>
    <!-- BPIC12W -->
    <tr><td rowspan="3">BPIC12W</td><td>3GD</td><td class="worst">0.794951</td><td>0.662948</td><td>0.631976</td><td class="best">0.350644</td></tr>
    <tr><td>CTD</td><td>191.924472</td><td>153.372272</td><td class="best">92.282846</td><td class="best">32.075031</td></tr>
    <tr><td>CAR</td><td class="best">28.736770</td><td class="worst">97.492991</td><td>92.738075</td><td class="prosit-col">79.711304</td></tr>
    <tr class="spacer-row"><td colspan="6"></td></tr>
    <!-- BPIC17W -->
    <tr><td rowspan="3">BPIC17W</td><td>3GD</td><td class="worst">0.814347</td><td>0.584121</td><td>0.581658</td><td class="best">0.319168</td></tr>
    <tr><td>CTD</td><td class="worst">148.790470</td><td>114.314488</td><td class="best">36.529998</td><td class="prosit-col">54.971774</td></tr>
    <tr><td>CAR</td><td class="worst">145.961497</td><td class="best">24.326364</td><td>82.289713</td><td class="prosit-col">53.717958</td></tr>
  </tbody>
</table>
</body>
</html>

**For more detailed analysis, please check the [Analysis Notebook](analysis.ipynb).**
