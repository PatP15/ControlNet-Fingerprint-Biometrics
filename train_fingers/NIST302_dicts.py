# Description: This file contains the dictionaries used to map the NIST302 data to the corresponding finger and technology.

id_to_finger = {
    '01': 'right thumb',
    '02': 'right index',
    '03': 'right middle',
    '04': 'right ring',
    '05': 'right pinky',
    '06': 'left thumb',
    '07': 'left index',
    '08': 'left middle',
    '09': 'left ring',
    '10': 'left pinky'
}

technology_mapping = {
    # From the first table (Challenger Branding Technology Prototype)
    "A": "Touch-free",
    "B": "Touch-free",
    "C": "Optical",
    "D": "Touch-free",
    "E": "Solid-state",
    "F": "Touch-free",
    "G": "Solid-state",
    "H": "Touch-free",
    # From the second table (Operator Branding Technology Data)
    "R": "Optical",  # Crossmatch L SCAN 1000PX
    "S": "Optical",  # Crossmatch Guardian USB
    "U": "Optical",  # Crossmatch L SCAN 1000PX (Rolled)
    "V": "Optical",  # Crossmatch L SCAN 1000PX (Rolled)
    # From the third table (Branding/Description Technology Data)
    "J": "Optical",  # Morpho TouchPrint 5300
    "K": "Optical",  # Michigan State University RaspiReader
    "L": "Touch-free",  # Unprocessed Captures from B (AOS ANDI N2N)
    "M": "Solid-state",  # Crossmatch EikonTouch 710
    "N": "Optical",  # Green Bit MultiScan 527g
    "P": "Optical",  # Futronic FS88
    "Q": "Optical",  # Crossmatch L SCAN 1000PX
    "T": "Touch-free",  # Unprocessed Captures from H (Clarkson University)
}

