from Functions import *

def main():
    api_key = 'live_05bd0dcf53fd4d62aa81405a08c117d9'
    project = "Traffic Sign Detection"
    tasks = get_API_tasks(api_key, project)

    audit_task(tasks, dilation_=True, plot=False)

if __name__ == "__main__":
    main()

