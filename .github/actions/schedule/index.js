const core = require('@actions/core');
const github = require('@actions/github');
const { Octokit } = require("@octokit/action");
const octokit = new Octokit();

const start = async function (a, b) {
    jobs = await octokit.request('GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs', {
        owner: 'Oneflow-Inc',
        repo: 'oneflow',
        run_id: 1976159286
    })
}

try {
    start()
} catch (error) {
    core.setFailed(error.message);
}
