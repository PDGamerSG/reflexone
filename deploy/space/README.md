---
title: ReflexOne
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: Locally trained typed text decisions across four use cases
---

# ReflexOne

Public API demonstration of [PDGamerSG/reflexone](https://github.com/PDGamerSG/reflexone).

Open the app to use the interactive API documentation. Choose `support` with the
`banking` or `hotels` domain, `commands` for 150-intent routing, or `spam` for SMS
classification. The service does not send messages, move money, or change bookings.

The support model is a fine-tuned six-layer MiniLM with 62 multilabel intent
outputs and 17 overlapping BIO entity streams. Commands use TF-IDF logistic
regression with calibrated rejection; spam uses TF-IDF Complement Naive Bayes.
Models run locally on CPU; no external inference API supplies predictions.

The image downloads the pinned, checksum-verified v0.1.0 artifacts during build.
Models are loaded once and readiness requires all three profiles. Free CPU Spaces
can sleep when idle, so the first request may need time to wake the app.

Read the [learning guide](https://github.com/PDGamerSG/reflexone/blob/main/docs/learning-guide.md),
[experiment report](https://github.com/PDGamerSG/reflexone/blob/main/docs/experiments.md),
and [model card](https://github.com/PDGamerSG/reflexone/blob/main/docs/model-card.md)
for training, measured performance, and limitations. Typed outputs can still be
semantically wrong. This demonstration has no availability or correctness guarantee.

Use public sample text in this public demonstration. Readiness is at `/ready`,
profile information at `/profiles`, and interactive documentation at `/docs`.
