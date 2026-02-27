import os

# Database
DB_PATH = 'data/arxiv.db'
EMBEDDING_MODEL_NAME = 'allenai/specter2_base'
SEMANTIC_SCHOLAR_API_KEY = 'RGbohKzzwD6PtmWhxZnNd8OgXxCj7byo8i7OSa8T'

# ArXiv API Defaults
DEFAULT_FETCH_FREQUENCY = 'daily' # daily, weekly
DEFAULT_TARGET_SUBJECTS = ['physics.comp-ph', 'physics.flu-dyn']

# Networking
LAN_BIND_HOST = '0.0.0.0'
PORT = 5001

# Mapping of ArXiv Codes to Human Readable Names
SUBJECT_CHOICES = {
    "ArXiv - Computer Science": {
        "cs.AI": "Artificial Intelligence", "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity", "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry", "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security", "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society", "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing", "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics", "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies", "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature", "cs.GR": "Graphics", "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction", "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory", "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science", "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia", "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis", "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture", "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems", "cs.PF": "Performance", "cs.PL": "Programming Languages",
        "cs.RO": "Robotics", "cs.SC": "Symbolic Computation", "cs.SD": "Sound",
        "cs.SE": "Software Engineering", "cs.SI": "Social and Information Networks", "cs.SY": "Systems and Control"
    },
    "ArXiv - Mathematics": {
        "math.AC": "Commutative Algebra", "math.AG": "Algebraic Geometry", "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology", "math.CA": "Classical Analysis and ODEs", "math.CB": "Classical Analysis and ODEs (secondary)",
        "math.CO": "Combinatorics", "math.CT": "Category Theory", "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry", "math.DS": "Dynamical Systems", "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics", "math.GN": "General Topology", "math.GR": "Group Theory",
        "math.GT": "Geometric Topology", "math.HO": "History and Overview", "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology", "math.LO": "Logic", "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics", "math.NA": "Numerical Analysis", "math.NT": "Number Theory",
        "math.OA": "Operator Algebras", "math.OC": "Optimization and Control", "math.PR": "Probability",
        "math.QA": "Quantum Algebra", "math.RA": "Rings and Algebras", "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry", "math.SP": "Spectral Theory", "math.ST": "Statistics Theory"
    },
    "ArXiv - Physics": {
        "astro-ph": "Astrophysics", "cond-mat": "Condensed Matter", "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment", "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology", "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics", "nlin": "Nonlinear Sciences", "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory", "physics": "Physics", "quant-ph": "Quantum Physics"
    },
    "ArXiv - Applied Physics & Sciences": {
        "physics.acc-ph": "Accelerator Physics", "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics", "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics", "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics", "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics", "physics.data-an": "Data Analysis, Statistics and Probability",
        "physics.ed-ph": "Physics Education", "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics", "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics", "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics", "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics", "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics and Society", "physics.space-ph": "Space Physics"
    },
    "ArXiv - Quantitative Biology": {
        "q-bio.BM": "Biomolecules", "q-bio.CB": "Cell Behavior", "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks", "q-bio.NC": "Neurons and Cognition", "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution", "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes", "q-bio.TO": "Tissues and Organs"
    },
    "ArXiv - Quantitative Finance": {
        "q-fin.CP": "Computational Finance", "q-fin.EC": "Economics", "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance", "q-fin.PM": "Portfolio Management", "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management", "q-fin.ST": "Statistical Finance", "q-fin.TR": "Trading and Market Microstructure"
    },
    "ArXiv - Statistics": {
        "stat.AP": "Applications", "stat.CO": "Computation", "stat.ME": "Methodology",
        "stat.ML": "Machine Learning", "stat.OT": "Other Statistics", "stat.TH": "Statistics Theory"
    },
    "ArXiv - Economics & Eng": {
        "econ.EM": "Econometrics", "econ.GN": "General Economics", "econ.TH": "Theoretical Economics",
        "eess.AS": "Audio and Speech Processing", "eess.IV": "Image and Video Processing", "eess.SP": "Signal Processing", "eess.SY": "Systems and Control"
    },
    "Biology (BioRxiv)": {
        "biorxiv.animal_behavior": "Animal Behavior", "biorxiv.bioinformatics": "Bioinformatics",
        "biorxiv.biophysics": "Biophysics", "biorxiv.cancer_biology": "Cancer Biology",
        "biorxiv.cell_biology": "Cell Biology", "biorxiv.clinical_trials": "Clinical Trials",
        "biorxiv.developmental_biology": "Developmental Biology", "biorxiv.ecology": "Ecology",
        "biorxiv.epidemiology": "Epidemiology", "biorxiv.evolutionary_biology": "Evolutionary Biology",
        "biorxiv.genetics": "Genetics", "biorxiv.genomics": "Genomics", "biorxiv.immunology": "Immunology",
        "biorxiv.microbiology": "Microbiology", "biorxiv.molecular_biology": "Molecular Biology",
        "biorxiv.neuroscience": "Neuroscience", "biorxiv.paleontology": "Paleontology",
        "biorxiv.pathology": "Pathology", "biorxiv.pharmacology": "Pharmacology",
        "biorxiv.physiology": "Physiology", "biorxiv.plant_biology": "Plant Biology",
        "biorxiv.scientific_communication": "Scientific Communication", "biorxiv.synthetic_biology": "Synthetic Biology",
        "biorxiv.systems_biology": "Systems Biology", "biorxiv.zoology": "Zoology"
    }
}
