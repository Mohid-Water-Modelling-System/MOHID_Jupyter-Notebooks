import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

def update_parameter_definitions(xml_file_path,Start,End,Integrator,Threads,OutputWriteTime,BufferSize):
    """
    Updates only the content of the <parameters> block in the XML file with simulation information,
    preserving any text outside this block unchanged.
    """
    """
        <parameters>
            <parameter key="Start" value="2024 01 24 00 00 00" comment="Date of initial instant" units_comment="space delimited ISO 8601 format up to seconds" />
            <parameter key="End"   value="2024 01 25 00 00 00" comment="Date of final instant" units_comment="ISO format" />
            <parameter key="Integrator" value="2" comment="Integration Algorithm 1:Euler, 2:Multi-Step Euler, 3:RK4 (default=1)" />			
            <parameter key="Threads" value="10" comment="Computation threads for shared memory computation (default=auto)" />
            <parameter key="OutputWriteTime" value="86400" comment="Time out data (1/Hz)" units_comment="seconds" />
            <parameter key="BufferSize" value="86400" comment="Optional parameter. Controls input frequency" units_comment="seconds" />
        </parameters>
    """      

    # Load the file into ElementTree (only to build our new block)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find the existing <simulation> element, or create one if missing.
    parameters_elem = root.find('.//parameters')
    if parameters_elem is None:
        parameters_elem = ET.Element('parameters')
    else:
        # Clear existing content if already present
        for child in list(parameters_elem):
            parameters_elem.remove(child)
    
    # Build new child elements for simulation details
    # Start parameter
    start_param = ET.Element("parameter")
    start_param.set("key", "Start")
    start_param.set("value", Start.strftime("%Y %m %d %H %M %S"))
    start_param.set("comment", "Date of initial instant")
    start_param.set("units_comment", "space delimited ISO 8601 format up to seconds")
    parameters_elem.append(start_param)

    # End parameter
    end_param = ET.Element("parameter")
    end_param.set("key", "End")
    end_param.set("value", End.strftime("%Y %m %d %H %M %S"))
    end_param.set("comment", "Date of final instant")
    end_param.set("units_comment", "ISO format")
    parameters_elem.append(end_param)

    # Integrator parameter
    integrator_param = ET.Element("parameter")
    integrator_param.set("key", "Integrator")
    integrator_param.set("value", str(Integrator))
    integrator_param.set("comment", "Integration Algorithm 1:Euler, 2:Multi-Step Euler, 3:RK4 (default=1)")
    parameters_elem.append(integrator_param)

    # Threads parameter
    threads_param = ET.Element("parameter")
    threads_param.set("key", "Threads")
    threads_param.set("value", str(Threads))
    threads_param.set("comment", "Computation threads for shared memory computation (default=auto)")
    parameters_elem.append(threads_param)

    # OutputWriteTime parameter
    output_param = ET.Element("parameter")
    output_param.set("key", "OutputWriteTime")
    output_param.set("value", str(OutputWriteTime))
    output_param.set("comment", "Time out data (1/Hz)")
    output_param.set("units_comment", "seconds")
    parameters_elem.append(output_param)

    # BufferSize parameter
    buffer_param = ET.Element("parameter")
    buffer_param.set("key", "BufferSize")
    buffer_param.set("value", str(BufferSize))
    buffer_param.set("comment", "Optional parameter. Controls input frequency")
    buffer_param.set("units_comment", "seconds")
    parameters_elem.append(buffer_param)

    
    # Generate a pretty-printed XML string for the new <simulation> block
    rough_string = ET.tostring(parameters_elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_parameters_elem = reparsed.toprettyxml(indent="    ")

    # Remove XML declaration and extraneous blank lines
    lines = pretty_parameters_elem.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]

    # Read the original file's XML to determine the current block's indent (if any)
    with open(xml_file_path, "r", encoding="utf-8") as f:
        original_xml = f.read()
    indent_match = re.search(r'(^\s*)<parameters\b', original_xml, re.MULTILINE)
    base_indent = indent_match.group(1) if indent_match else ""

    # Reindent the block:
    # The first line (<simulation>) is made flush left by stripping all leading whitespace.
    reindented_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            reindented_lines.append(line.lstrip())
        else:
            reindented_lines.append(base_indent + line)
    reindented_block = "\n".join(reindented_lines)

    # Replace only the <simulation> block in the original XML
    pattern = re.compile(r'(<parameters\b[^>]*>).*?(</parameters>)', re.DOTALL)
    new_xml = pattern.sub(reindented_block, original_xml)

    # Write the updated XML back to the file
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write(new_xml)

    print(f"Updated <parameters> block in '{xml_file_path}' successfully.")
    
def update_simulation_definitions(xml_file_path,resolution,timestep,BoundingBoxMin,BoundingBoxMax,VerticalVelMethod,BathyminNetcdf,RemoveLandTracer,TracerMaxAge):
    """
        Updates only the content of the <simulation> block in the XML file with simulation information,
        preserving any text outside this block unchanged.

        <simulation>
                <resolution dp="8000" units_comment="metres (m)"/>
                <timestep dt="180" units_comment="seconds (s)"/>
                <BoundingBoxMin x="-42.269587" y="-11.61052" z="-1" units_comment="(deg,deg,m)"/>
                <BoundingBoxMax x="-22.416399" y="4.476972" z="1" units_comment="(deg,deg,m)"/>	
                <VerticalVelMethod value="3" comment="1:From velocity fields, 2:Divergence based, 3:Disabled. Default = 1" />
                <BathyminNetcdf value="0" comment="bathymetry is a property in the netcdf. 1:true, 0:false (computes from layer depth and openPoints. Default = 1"/>
                <RemoveLandTracer value="0" comment="Remove tracers on land 0:No, 1:Yes. Default = 1" />
                <TracerMaxAge value="0" comment="maximum tracer age. Default = 0.0. read if > 0" />
            </simulation>
        The new block is pretty-printed (with line breaks and indentations) and then re-indented to match the original file.
    """

    # Load the file into ElementTree (only to build our new block)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find the existing <simulation> element, or create one if missing.
    simulation_elem = root.find('.//simulation')
    if simulation_elem is None:
        simulation_elem = ET.Element('simulation')
    else:
        # Clear existing content if already present
        for child in list(simulation_elem):
            simulation_elem.remove(child)
    
    # Build new child elements for simulation details
    resolution_elem = ET.Element("resolution")
    resolution_elem.set("dp", str(resolution))
    resolution_elem.set("units_comment", "metres (m)")
    simulation_elem.append(resolution_elem)

    timestep_elem = ET.Element("timestep")
    timestep_elem.set("dt", str(timestep))
    timestep_elem.set("units_comment", "seconds (s)")
    simulation_elem.append(timestep_elem)

    BoundingBoxMin_elem = ET.Element("BoundingBoxMin")
    BoundingBoxMin_elem.set("x", f"{BoundingBoxMin[0]:.5f}")
    BoundingBoxMin_elem.set("y", f"{BoundingBoxMin[1]:.5f}")
    BoundingBoxMin_elem.set("z", f"{BoundingBoxMin[2]:.5f}")
    BoundingBoxMin_elem.set("units_comment", "(deg,deg,m)")
    simulation_elem.append(BoundingBoxMin_elem)

    BoundingBoxMax_elem = ET.Element("BoundingBoxMax")
    BoundingBoxMax_elem.set("x", f"{BoundingBoxMax[0]:.5f}")
    BoundingBoxMax_elem.set("y", f"{BoundingBoxMax[1]:.5f}")
    BoundingBoxMax_elem.set("z", f"{BoundingBoxMax[2]:.5f}")
    BoundingBoxMax_elem.set("units_comment", "(deg,deg,m)")
    simulation_elem.append(BoundingBoxMax_elem)

    VerticalVelMethod_elem = ET.Element("VerticalVelMethod")
    VerticalVelMethod_elem.set("value", str(VerticalVelMethod))
    VerticalVelMethod_elem.set("comment", "1:From velocity fields, 2:Divergence based, 3:Disabled. Default = 1")
    simulation_elem.append(VerticalVelMethod_elem)

    BathyminNetcdf_elem = ET.Element("BathyminNetcdf")
    BathyminNetcdf_elem.set("value", str(BathyminNetcdf))
    BathyminNetcdf_elem.set("comment", "bathymetry is a property in the netcdf. 1:true, 0:false (computes from layer depth and openPoints). Default = 1")
    simulation_elem.append(BathyminNetcdf_elem)

    RemoveLandTracer_elem = ET.Element("RemoveLandTracer")
    RemoveLandTracer_elem.set("value", str(RemoveLandTracer))
    RemoveLandTracer_elem.set("comment", "Remove tracers on land 0:No, 1:Yes. Default = 1")
    simulation_elem.append(RemoveLandTracer_elem)

    TracerMaxAge_elem = ET.Element("TracerMaxAge")
    TracerMaxAge_elem.set("value", str(TracerMaxAge))
    TracerMaxAge_elem.set("comment", "maximum tracer age. Default = 0.0. read if > 0")
    simulation_elem.append(TracerMaxAge_elem)

    # Generate a pretty-printed XML string for the new <simulation> block
    rough_string = ET.tostring(simulation_elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_simulation_elem = reparsed.toprettyxml(indent="    ")

    # Remove XML declaration and extraneous blank lines
    lines = pretty_simulation_elem.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]

    # Read the original file's XML to determine the current block's indent (if any)
    with open(xml_file_path, "r", encoding="utf-8") as f:
        original_xml = f.read()
    indent_match = re.search(r'(^\s*)<simulation\b', original_xml, re.MULTILINE)
    base_indent = indent_match.group(1) if indent_match else ""

    # Reindent the block:
    # The first line (<simulation>) is made flush left by stripping all leading whitespace.
    reindented_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            reindented_lines.append(line.lstrip())
        else:
            reindented_lines.append(base_indent + line)
    reindented_block = "\n".join(reindented_lines)

    # Replace only the <simulation> block in the original XML
    pattern = re.compile(r'(<simulation\b[^>]*>).*?(</simulation>)', re.DOTALL)
    new_xml = pattern.sub(reindented_block, original_xml)

    # Write the updated XML back to the file
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write(new_xml)

    print(f"Updated <simulation> block in '{xml_file_path}' successfully.")

def update_source_definitions(xml_file_path, markers_dict,rate_seconds,rate_trcPerEmission):
    """
    Updates only the content of the <sourceDefinitions> block in the XML file with marker information,
    preserving any text outside this block unchanged.
    
    For each marker in markers_dict (structure: { marker_id: {'location': [lon, lat], 'name': marker_name} }),
    the following XML structure is created:
    
    <source>
        <setsource id="marker_id" name="marker_name" />
        <rate_seconds value="21600" comment="emission step in seconds. 3600 is a tracer per hour" />
        <rate_trcPerEmission value="1" comment="number of tracers emiited every rate_seconds. 5 is a 5 tracers per rate_seconds" />
        <point x="lon_value" y="lat_value" z="0" units_comment="(deg,deg,m)"/>
    </source>
    
    The new block is pretty-printed (with line breaks and indentations) and then re-indented to match the original file.
    """
    # -----------------------------
    # Build a new <sourceDefinitions> subtree
    # -----------------------------
    # Load the file into ElementTree (only to build our new block)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find the existing <sourceDefinitions> element, or create one if missing.
    source_defs = root.find('.//sourceDefinitions')
    if source_defs is None:
        source_defs = ET.Element('sourceDefinitions')
    else:
        # Clear existing content
        for child in list(source_defs):
            source_defs.remove(child)
    
    # Create a new <source> element for each marker.
    for marker_id, marker_data in markers_dict.items():
        location = marker_data.get("location", [1, 0])  # Expected format: [lon, lat]
        marker_name = marker_data.get("name", f"Marker {marker_id}")
        
        source_elem = ET.Element("source")
        
        # <setsource> element with marker id and name
        setsource_elem = ET.SubElement(source_elem, "setsource")
        setsource_elem.set("id", str(marker_id))
        setsource_elem.set("name", marker_name)
        
        # <rate_seconds> element (using constant values)
        rate_sec_elem = ET.SubElement(source_elem, "rate_seconds")
        rate_sec_elem.set("value", str(rate_seconds))
        rate_sec_elem.set("comment", "emission step in seconds. 3600 is a tracer per hour")
        
        # <rate_trcPerEmission> element (using constant values)
        rate_trc_elem = ET.SubElement(source_elem, "rate_trcPerEmission")
        rate_trc_elem.set("value", str(rate_trcPerEmission))
        rate_trc_elem.set("comment", "number of tracers emited every rate_seconds. 5 is a 5 tracers per rate_seconds")
        
        # <point> element with coordinate information.
        point_elem = ET.SubElement(source_elem, "point")
        point_elem.set("x", f"{location[1]:.5f}")
        point_elem.set("y", f"{location[0]:.5f}")
        point_elem.set("z", "0")
        point_elem.set("units_comment", "(deg,deg,m)")
        
        # Append the marker element.
        source_defs.append(source_elem)
    
  # Generate a pretty-printed XML string for the new <sourceDefinitions> block.
    rough_string = ET.tostring(source_defs, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_source_defs = reparsed.toprettyxml(indent="    ")
    
    # Remove XML declaration if present and filter out any extraneous blank lines.
    lines = pretty_source_defs.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]
    # Re-join the lines (this string is used only for debugging, the list "lines" is used below)
    pretty_source_defs = "\n".join(lines)
    
    # -----------------------------
    # Re-indent the new block to match the existing block's indent
    # -----------------------------
    # Open the original XML file as text
    with open(xml_file_path, "r", encoding="utf-8") as f:
        original_xml = f.read()
    
    # Determine the existing <sourceDefinitions> tag's leading whitespace.
    indent_match = re.search(r'(^\s*)<sourceDefinitions\b', original_xml, re.MULTILINE)
    base_indent = indent_match.group(1) if indent_match else ""
    
    # Reindent the block:
    # The first line (<sourceDefinitions>) is forced flush left (using lstrip()),
    # while every subsequent line is prefixed with the original base_indent.
    reindented_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            reindented_lines.append(line.lstrip())
        else:
            reindented_lines.append(base_indent + line)
    reindented_block = "\n".join(reindented_lines)

    # -----------------------------
    # Replace only the <sourceDefinitions> block in the original file
    # -----------------------------
    pattern = re.compile(r'(<sourceDefinitions\b[^>]*>).*?(</sourceDefinitions>)', re.DOTALL)
    new_xml = pattern.sub(reindented_block, original_xml)
    
    # Write the updated XML back to the file
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write(new_xml)
    
    print(f"Updated <sourceDefinitions> block in '{xml_file_path}' with {len(markers_dict)} marker(s).")