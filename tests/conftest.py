import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)                                   # frozen fixture
sys.path.insert(0, os.path.join(HERE, '..', 'src'))        # ksz_pipeline
