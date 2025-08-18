from data_loader import load_data
from model import build_phase4_model as build_model
from results import dump_results  
import sys
from datetime import datetime
from gurobipy import GRB  
TIME_LIMIT_SEC = 10000      # max solver time

xls_path = "non_coop_case.xlsx"   

data  = load_data(xls_path)
model = build_model(data)

# ── Solver configuration ───────────────────────────────────────
model.Params.TimeLimit = TIME_LIMIT_SEC  
model.Params.OutputFlag = 1             

model.optimize()

if model.status == GRB.INFEASIBLE:
    print("Model is infeasible – computing IIS …")
    model.computeIIS()
    model.write("phase4_inf.ilp")     
    model.write("phase4_inf.json")    
    model.write("phase4_inf.mps")

    # --- quick console summary -------------------------------------
    print("\n──── IIS constraints ────")
    try:
        with open("phase4_inf.ilp") as fh:
            section = None
            for raw in fh:
                line = raw.strip()
                if line == "Subject To":
                    section = "constr"
                    continue
                if line == "Bounds":
                    section = "bounds"
                    print("\n──── IIS variable bounds ────")
                    continue
                if section == "constr" and line.endswith(":"):
                    print(line[:-1])
                elif section == "bounds" and line and not line.startswith(("Binary", "End", "Generals")):
                    print(line)
    except FileNotFoundError:
        print("IIS file phase4_inf.ilp not found.")

    # stop the script so downstream code doesn’t access .X attributes
    sys.exit(1)

# Check for incumbent solution (any feasible solution found)
if model.SolCount == 0:
    print("No feasible solution found within the time limit (status:", model.status, ")")
    sys.exit(0)



# ------------------------------------------------------------------
# Convenience aliases used in the reporting sections below
# ------------------------------------------------------------------
ports = data["ports"]
fuels = data["fuels"]
Tau   = data["op_steps"]

qDE = model._vars.get("qDE", {})  
u   = model._vars.get("u",   {})  
qCH = model._vars.get("qCH", {})  
qDC = model._vars.get("qDC", {})  
qGR = model._vars.get("qGR", {})  
e   = model._vars.get("e",   {})  
u   = model._vars.get("u",   {})  
qCH = model._vars.get("qCH", {})  
qDC = model._vars.get("qDC", {})  
qGR = model._vars.get("qGR", {})  
e   = model._vars.get("e",   {})  

                
timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

print("Solve status :", model.status, "(code", int(model.status), ")")
if model.status == GRB.TIME_LIMIT:
    print("⚠ Reached time limit; incumbent objective:", model.ObjVal)
else:
    print("Objective    :", model.ObjVal)

# ------------------------------------------------------------------
# Save results (always writes an Excel workbook, even on partial runs)
# ------------------------------------------------------------------
try:
    dump_results(model, tag=timestamp)          # results_<timestamp>.xlsx
    print(f"\n✔ Results exported to results_{timestamp}.xlsx")
except Exception as exc:
    print(f"\n⚠ Dumping results failed: {exc}")