import subprocess


def get_changed_files(base=None, head=None):
    changed = subprocess.check_output(
        f"git diff --name-only --diff-filter=ACMRT {base} {head}",
        shell=True,
        text=True,
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
        or f.endswith(".cmake")
        or f.endswith("CMakeLists.txt")
    ]
    print("[changed]", changed)
    print("[not_single_client_files]", not_single_client_files)
    return len(not_single_client_files) < len(changed)


def print_github_action_output(name=None, value=None):
    print(f"::set-output name={name}::{value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--need_single_client_tests", action="store_true")
    args = parser.parse_args()
    files = get_changed_files(base=args.base, head=args.head)
    if should_run_single_client_tests(changed=files) or args.need_single_client_tests:
        print_github_action_output(name="should_run_single_client_tests", value="1")
