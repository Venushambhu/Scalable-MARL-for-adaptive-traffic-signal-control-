import xml.etree.ElementTree as ET

tree1 = ET.parse("dynamic_ew.rou.xml")
tree2 = ET.parse("dynamic_ns.rou.xml")

root1 = tree1.getroot()
root2 = tree2.getroot()

# Collect all vehicle elements from both files
vehicles = list(root1.findall("vehicle")) + list(root2.findall("vehicle"))

# Sort by depart time (SUMO requires chronological order)
vehicles.sort(key=lambda v: float(v.get("depart")))

# Build merged output using root1's structure (routes/vTypes) as the base
merged_root = ET.Element("routes")
for child in root1:
    if child.tag != "vehicle":
        merged_root.append(child)

for v in vehicles:
    merged_root.append(v)

merged_tree = ET.ElementTree(merged_root)
ET.indent(merged_tree, space="    ")
merged_tree.write("dynamic_demand.rou.xml", xml_declaration=True, encoding="UTF-8")

print(f"Merged {len(vehicles)} vehicles into dynamic_demand.rou.xml")