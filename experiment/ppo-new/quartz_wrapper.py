from typing import Any, Callable, Dict, List, Set, Tuple

import sys
sys.path.insert(0, '../..')
import quartz  # type: ignore

"""global vars"""
quartz_context: quartz.QuartzContext
quartz_parser: quartz.PyQASMParser

has_parameterized_gate: bool


def ibm_add_xfer(context: quartz.QuartzContext):
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
        context.add_xfer_from_qasm_str(src_str=circ_pair[0], dst_str=circ_pair[1])
        context.add_xfer_from_qasm_str(src_str=circ_pair[1], dst_str=circ_pair[0])

    return context


def init_quartz_context(
    gate_set: List[str],
    ecc_file_path: str,
    no_increase: bool,
    include_nop: bool,
) -> None:
    """
    Initialize Quartz context and parser.
    
    Args:
        gate_set: List of gate type strings (e.g., ["h", "cx", "rz"])
        ecc_file_path: Path to ECC set file
        no_increase: Whether to disable cost increase
        include_nop: Whether to include NOP transformation
    """
    global quartz_context
    global quartz_parser
    global has_parameterized_gate
    
    try:
        # Try the original API first
        quartz_context = quartz.QuartzContext(
            gate_set=gate_set,
            filename=ecc_file_path,
            no_increase=no_increase,
            include_nop=include_nop,
        )
    except (TypeError, AttributeError) as e:
        print(f"Original QuartzContext constructor failed: {e}")
        print("Trying alternative initialization method...")
        
        # Fallback: convert gate_set strings to GateType enums
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
            if gate.lower() in gate_type_map:
                gate_types.append(gate_type_map[gate.lower()])
            else:
                print(f"Warning: Unknown gate type '{gate}'")
        
        # Add input types if not present
        if quartz.GateType.input_qubit not in gate_types:
            gate_types.append(quartz.GateType.input_qubit)
        if quartz.GateType.input_param not in gate_types:
            gate_types.append(quartz.GateType.input_param)
        
        # Create context with gate types
        param_info = quartz.ParamInfo()
        quartz_context = quartz.Context(gate_types, param_info)
        
        # Load ECC set
        equiv_set = quartz.EquivalenceSet()
        if not equiv_set.load_json(quartz_context, ecc_file_path):
            raise ValueError(f"Failed to load ECC set from {ecc_file_path}")
        
        # Attach equivalence set to context
        quartz_context._equiv_set = equiv_set
        
        # Add missing methods if needed
        _add_context_methods()
    
    try:
        quartz_parser = quartz.PyQASMParser(context=quartz_context)
    except (TypeError, AttributeError):
        # Fallback to regular QASMParser
        quartz_parser = quartz.QASMParser(quartz_context)
    
    try:
        has_parameterized_gate = quartz_context.has_parameterized_gate()
    except AttributeError:
        # Manually check for parameterized gates
        parameterized_gates = {'rx', 'ry', 'rz', 'u1', 'u2', 'u3'}
        has_parameterized_gate = any(gate.lower() in parameterized_gates for gate in gate_set)
    
    # Add IBM-specific transformations if needed
    if 'ibm_325_ecc' in ecc_file_path:
        try:
            quartz_context = ibm_add_xfer(quartz_context)
        except Exception as e:
            print(f"Warning: Could not add IBM xfers: {e}")


def qasm_to_graph_th_dag(qasm_str: str) -> quartz.PyGraph:
    """
    Convert QASM string to PyGraph using DAG intermediate representation.
    
    Args:
        qasm_str: QASM code as string
        
    Returns:
        quartz.PyGraph object
    """
    global quartz_context
    global quartz_parser
    
    try:
        dag = quartz_parser.load_qasm_str(qasm_str)
        graph = quartz.PyGraph(context=quartz_context, dag=dag)
        return graph
    except AttributeError:
        # Fallback: use alternative method
        return qasm_to_graph(qasm_str)


def qasm_to_graph(qasm_str: str) -> quartz.PyGraph:
    """
    Convert QASM string to PyGraph.
    
    Args:
        qasm_str: QASM code as string
        
    Returns:
        quartz.PyGraph object
    """
    global quartz_context
    
    try:
        # Try the direct method first
        graph = quartz.PyGraph.from_qasm_str(context=quartz_context, qasm_str=qasm_str)
        return graph
    except AttributeError:
        # Fallback: use temp file method
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


def is_nop(xfer_id: int) -> bool:
    """
    Check if a transformation is a NOP (no-operation).
    
    Args:
        xfer_id: Transformation ID
        
    Returns:
        True if the transformation is NOP, False otherwise
    """
    global quartz_context
    
    try:
        xfer = quartz_context.get_xfer_from_id(id=xfer_id)
        return xfer.is_nop
    except AttributeError:
        # Fallback: return False if we can't determine
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
                        return eqs[id]
                except Exception as e:
                    raise RuntimeError(f"Could not get xfer {id}: {e}")
            raise RuntimeError("Equivalence set not loaded")
        
        quartz.Context.get_xfer_from_id = get_xfer_from_id
