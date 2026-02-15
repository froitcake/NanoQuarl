from typing import Any, Callable, Dict, List, Set, Tuple

import sys
sys.path.insert(0, '../..')
import quartz  # type: ignore

"""global vars"""
quartz_context = None
quartz_parser = None
has_parameterized_gate: bool = False


def ibm_add_xfer(context):
    """Add IBM-specific transformations"""
    equivalent_circ_pairs = [
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi) q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi) q[0]; sx q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(pi/2) q[0]; sx q[0]; rz(pi/2) q[0];',
        ),
        (
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0]; rz(3*pi/2) q[0]; sx q[0];',
            'OpenQASM 2.0; include "qelib1.inc"; qreg q[1]; rz(3*pi/2) q[0]; sx q[0]; rz(3*pi/2) q[0];',
        ),
    ]

    for circ_pair in equivalent_circ_pairs:
        print(circ_pair)
        try:
            context.add_xfer_from_qasm_str(src_str=circ_pair[0], dst_str=circ_pair[1])
            context.add_xfer_from_qasm_str(src_str=circ_pair[1], dst_str=circ_pair[0])
        except Exception as e:
            print(f"Warning: Could not add IBM xfer pair: {e}")

    return context


def init_quartz_context(
    gate_set: List[str],
    ecc_file_path: str,
    no_increase: bool,
    include_nop: bool,
) -> None:
    """
    Initialize Quartz context and parser using the working fallback approach.
    
    Args:
        gate_set: List of gate type strings (e.g., ["h", "cx", "rz"])
        ecc_file_path: Path to ECC set file
        no_increase: Whether to disable cost increase
        include_nop: Whether to include NOP transformation
    """
    global quartz_context
    global quartz_parser
    global has_parameterized_gate
    
    print(f"Initializing Quartz context with gate_set: {gate_set}")
    print(f"Loading ECC file: {ecc_file_path}")
    
    # Map gate strings to GateType enums
    gate_type_map = {
        'h': quartz.GateType.h,
        'x': quartz.GateType.x,
        'y': quartz.GateType.y,
        'z': quartz.GateType.z,
        'cx': quartz.GateType.cx,
        'cy': quartz.GateType.cy,
        'cz': quartz.GateType.cz,
        'ccx': quartz.GateType.ccx,
        'ccz': quartz.GateType.ccz,
        'rx': quartz.GateType.rx,
        'ry': quartz.GateType.ry,
        'rz': quartz.GateType.rz,
        'u1': quartz.GateType.u1,
        'u2': quartz.GateType.u2,
        'u3': quartz.GateType.u3,
        'sx': quartz.GateType.sx,
        'add': quartz.GateType.add,
        'input_qubit': quartz.GateType.input_qubit,
        'input_param': quartz.GateType.input_param,
    }
    
    gate_types = []
    for gate in gate_set:
        gate_lower = gate.lower()
        if gate_lower in gate_type_map:
            gate_types.append(gate_type_map[gate_lower])
        else:
            print(f"Warning: Unknown gate type '{gate}', skipping")
    
    # Add input types if not present
    if quartz.GateType.input_qubit not in gate_types:
        gate_types.append(quartz.GateType.input_qubit)
    if quartz.GateType.input_param not in gate_types:
        gate_types.append(quartz.GateType.input_param)
    
    print(f"Creating context with {len(gate_types)} gate types")
    
    # Create context with gate types
    param_info = quartz.ParamInfo()
    quartz_context = quartz.Context(gate_types, param_info)
    
    # Load ECC set
    equiv_set = quartz.EquivalenceSet()
    print(f"Loading equivalence set from {ecc_file_path}...")
    if not equiv_set.load_json(quartz_context, ecc_file_path):
        raise ValueError(f"Failed to load ECC set from {ecc_file_path}")
    
    print(f"Successfully loaded equivalence set")
    
    # Attach equivalence set to context
    quartz_context._equiv_set = equiv_set
    
    # Add helper methods to context
    _add_context_methods()
    
    # Create parser
    try:
        quartz_parser = quartz.QASMParser(quartz_context)
        print("Created QASMParser successfully")
    except Exception as e:
        print(f"Warning: Could not create QASMParser: {e}")
        quartz_parser = None
    
    # Check for parameterized gates
    parameterized_gates = {'rx', 'ry', 'rz', 'u1', 'u2', 'u3'}
    has_parameterized_gate = any(gate.lower() in parameterized_gates for gate in gate_set)
    
    # Add IBM-specific transformations if needed
    if 'ibm' in ecc_file_path.lower():
        print("Adding IBM-specific transformations...")
        try:
            quartz_context = ibm_add_xfer(quartz_context)
        except Exception as e:
            print(f"Warning: Could not add IBM xfers: {e}")
    
    print(f"Quartz context initialized with {quartz_context.num_xfers} transformations")


def qasm_to_graph_th_dag(qasm_str: str):
    """
    Convert QASM string to Graph using DAG intermediate representation.
    
    Args:
        qasm_str: QASM code as string
        
    Returns:
        quartz.Graph object
    """
    global quartz_context
    global quartz_parser
    
    if quartz_parser is None:
        raise RuntimeError("Quartz parser not initialized. Call init_quartz_context() first.")
    
    try:
        # Use temp file method for compatibility
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as f:
            f.write(qasm_str)
            temp_path = f.name
        
        try:
            circuit_seq = quartz_parser.load_qasm(temp_path)
            graph = quartz.Graph(quartz_context, circuit_seq)
            return graph
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        print(f"Error in qasm_to_graph_th_dag: {e}")
        raise


def qasm_to_graph(qasm_str: str):
    """
    Convert QASM string to Graph.
    
    Args:
        qasm_str: QASM code as string
        
    Returns:
        quartz.Graph object
    """
    global quartz_context
    global quartz_parser
    
    if quartz_context is None:
        raise RuntimeError("Quartz context not initialized. Call init_quartz_context() first.")
    
    # Use the same method as qasm_to_graph_th_dag for consistency
    return qasm_to_graph_th_dag(qasm_str)


def is_nop(xfer_id: int) -> bool:
    """
    Check if a transformation is a NOP (no-operation).
    
    Args:
        xfer_id: Transformation ID
        
    Returns:
        True if the transformation is NOP, False otherwise
    """
    global quartz_context
    
    if quartz_context is None:
        return False
    
    try:
        xfer = quartz_context.get_xfer_from_id(id=xfer_id)
        if hasattr(xfer, 'is_nop'):
            return xfer.is_nop
        return False
    except Exception:
        return False


def _add_context_methods():
    """Add methods to quartz.Context that may be missing"""
    
    if not hasattr(quartz.Context, 'num_xfers'):
        def num_xfers_property(self) -> int:
            """Get number of transformations from equivalence set"""
            if hasattr(self, '_equiv_set'):
                try:
                    if hasattr(self._equiv_set, 'size'):
                        return self._equiv_set.size()
                    elif hasattr(self._equiv_set, 'num_equivalences'):
                        return self._equiv_set.num_equivalences()
                    else:
                        eqs = self._equiv_set.get_all_equivalences()
                        return len(eqs)
                except Exception as e:
                    print(f"Warning: Could not get num_xfers: {e}")
                    return 0
            return 0
        
        quartz.Context.num_xfers = property(num_xfers_property)
    
    if not hasattr(quartz.Context, 'get_xfer_from_id'):
        def get_xfer_from_id(self, id: int):
            """Get transformation by ID"""
            if hasattr(self, '_equiv_set'):
                try:
                    if hasattr(self._equiv_set, 'get_equivalence'):
                        return self._equiv_set.get_equivalence(id)
                    elif hasattr(self._equiv_set, 'get_xfer'):
                        return self._equiv_set.get_xfer(id)
                    else:
                        eqs = self._equiv_set.get_all_equivalences()
                        if 0 <= id < len(eqs):
                            return eqs[id]
                        raise IndexError(f"Transformation ID {id} out of range")
                except Exception as e:
                    raise RuntimeError(f"Could not get xfer {id}: {e}")
            raise RuntimeError("Equivalence set not loaded")
        
        quartz.Context.get_xfer_from_id = get_xfer_from_id
    
    if not hasattr(quartz.Context, 'has_parameterized_gate'):
        def has_parameterized_gate(self) -> bool:
            """Check if context has parameterized gates"""
            global has_parameterized_gate
            return has_parameterized_gate
        
        quartz.Context.has_parameterized_gate = has_parameterized_gate
