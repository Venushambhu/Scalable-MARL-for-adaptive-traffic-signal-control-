import traci

traci.start(["sumo", "-n", "test.net.xml", "--no-warnings"])
traci.route.add("r0", ["AB"])
traci.vehicle.add("veh0", "r0")

for step in range(20):
    traci.simulationStep()
    if "veh0" in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition("veh0")
        print(f"step {step}: vehicle at {pos}")

traci.close()
print("TraCI test successful")
