const { Octokit } = require("@octokit/core");
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const owner = 'Oneflow-Inc';
const repo = 'oneflow';
const start = async function (a, b) {
    // await octokit.request('GET /repos/{owner}/{repo}/actions/workflows', {
    //     owner: owner,
    //     repo: repo
    // }).then(r => console.log(r.data.workflows))

    await octokit.request('GET /repos/{owner}/{repo}/actions/runs', {
        owner: owner,
        repo: repo,
        status: "in_progress"
    }).then(r => r.data.workflow_runs.map(wr => {
        console.log(wr.name)
        console.log(wr.jobs_url)
        wr.pull_requests.map(pr => console.log(pr.url))
    }))
}
start()
