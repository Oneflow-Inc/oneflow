import subprocess


def get_changed_files(base=None, head=None):
    changed = subprocess.check_output(
        "git diff --name-only --diff-filter=ACMRT master HEAD", shell=True, text=True
    )
    changed = str(changed).splitlines()
    return changed


def should_run_single_client_tests(changed=None):
    not_single_client_files = [
        f
        for f in changed
        if (
            f.endswith(".py")
            and not f.startswith("python/oneflow/compatible/single_client")
        )
        or f.endswith(".yml")
        or f.endswith(".rst")
        or f.endswith(".md")
    ]
    print("[changed]", not_single_client_files)
    print("[not_single_client_files]", not_single_client_files)
    return len(not_single_client_files) < len(changed)


def print_github_action_output(changed=None):
    print(should_run_single_client_tests(changed))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--head", type=str, required=True)
    args = parser.parse_args()
    files = get_changed_files(base=args.base, head=args.head)
    print_github_action_output(files)
