# Contributing to MusicVision

Contributions are welcome! By contributing to MusicVision, you agree to the terms below.

## Developer Certificate of Origin (DCO)

MusicVision uses a dual-license model (PolyForm Noncommercial + Commercial). To ensure that contributions can be distributed under both licenses, all contributors must certify their submissions under the [Developer Certificate of Origin v1.1](https://developercertificate.org/):

> Developer Certificate of Origin v1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
>
> (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
>
> (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

### How to sign off

Add a `Signed-off-by` line to your commit messages:

```
Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically with `git commit -s`.

### License grant for dual licensing

By submitting a contribution with a DCO sign-off, you additionally grant the project maintainer (Todd Green) a perpetual, irrevocable, worldwide, royalty-free license to use, modify, sublicense, and distribute your contribution under any license, including commercial licenses. This grant is necessary for the dual-license model to work — without it, commercial licenses could not cover contributed code.

If you are not comfortable with this grant, you are still free to use MusicVision under the PolyForm Noncommercial license, but please do not submit contributions.

## Guidelines

- Run `pytest tests/ -v --tb=short` before submitting a PR
- Run `ruff check src/ tests/` and fix any lint errors
- Follow the coding conventions in [CLAUDE.md](CLAUDE.md)
- Add tests for new functionality
- Keep PRs focused — one feature or fix per PR

## What to contribute

- Bug fixes
- New engine integrations
- Documentation improvements
- Test coverage
- Performance improvements

For larger features, please open an issue first to discuss the approach.
