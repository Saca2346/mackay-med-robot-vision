"""Launcher for aipc_tasks.py with --procs N: spawned OCR processes import this tiny file as their main module, not
aipc_tasks.py (which loads the detector and the pools). Usage: python aipc_main.py <aipc_tasks arguments>"""
if __name__ == "__main__":
    import aipc_tasks
    aipc_tasks.main()
