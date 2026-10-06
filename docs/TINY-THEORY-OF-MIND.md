# Tiny Theory of Mind: project fit

Reviewed October 6, 2026 against the public
[dataset card and examples](https://huggingface.co/datasets/AxiomicLabs/Tiny_Theory_of_Mind).
This is a synthetic text benchmark for small language models, with four-choice
continuation scoring. It is not an agent architecture or a memory component.
Only public information was inspected; no dataset artifact was downloaded,
model trained or project data sent to the publisher.

Laya and Clef could be evaluated through a separately tested text-choice
adapter with input-length and option-order checks. Adapter-based scores would
need their own label and should not be equated with the publisher's scoring
method without reproducing it. Preserve evaluation items: training on them and
reporting the same items as unseen would obscure the result.

The current PPO residents take spatial and bodily observations, not English
stories. The useful connection to GT-05 is to design embodied situations with
different private knowledge: for example, one resident sees a resource move
while another is absent. Measure whether learned signals help the uninformed
resident on new maps, with signal-removal controls. Such worlds can test social
reasoning without prescribing an agent's response or assigning tone meanings.

Recommendation: retain this as a later social-evaluation candidate while the
authorized GT-01 feeding and experimental restart work continues. It does not
establish that memory capacity is the present bottleneck, and a text benchmark
score alone would not prove useful communication inside the game.
