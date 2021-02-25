const { Octokit } = require("@octokit/core");
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const owner = 'Oneflow-Inc';
const repo = 'oneflow';
const should_start = async function (a, b) {
    runs = await octokit.request('GET /repos/{owner}/{repo}/actions/runs', {
        owner: owner,
        repo: repo,
        status: "in_progress"
    })
        .then(r =>
            r.data.workflow_runs
        )
    promises = runs.map(async wr => {
        wr.pull_requests.map(pr => console.log(pr.url))
        const r = await octokit.request('GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs', {
            owner: owner,
            repo: repo,
            run_id: wr.id
        });
        return r.data.jobs;
    })
    jobs_list = await Promise.all(promises)
    num_queued_jobs = jobs_list
        .map(jobs => {
            queued_jobs = jobs
                .filter(j => j.status == "queued")
            if (queued_jobs.length != 0) {
                queued_jobs.map(j => console.log(j.name, "||", j.status))
            }
            return queued_jobs.length
        }).reduce((a, b) => a + b, 0)
    console.log(num_queued_jobs)
    return num_queued_jobs == 0
}
let delay = async (seconds) => {
    setTimeout(() => console.log("after 1s"), 1000 * seconds)
}

async function start() {
    let i = 0;
    while (i < 1000) {
        if (await should_start()) {
            break;
        }
        await delay(60)
        i++;
    }
}

start().catch(console.error)
