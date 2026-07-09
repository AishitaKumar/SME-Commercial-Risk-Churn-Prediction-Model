# Who Do You Act On First? Building a Brain for Customer Retention

When I started this project, I thought the hard part would be predicting who's about
to leave. It wasn't. The hard part was the question that comes *after* the prediction:
okay, you know who might leave — now who do you actually act on first, and how?

Let me show you what I mean. Picture yourself running the retention team at a credit
card company. It's Monday morning. You've got a list of 10,000 members, a small team
that can only run so many interventions a month, and a quiet, expensive problem: some of these
people are about to walk, and you have no idea which ones.

And here's the part that keeps you up at night — every move has a cost. Reach out to the
wrong person, and you've spent money on someone who was never leaving. Miss the right
one, and a valuable customer is gone. Chase someone who'd already checked out months
ago, and you've just annoyed them on their way out the door.

So "who might churn?" isn't really your question. Your question is sharper:

**Given a limited team and a limited budget, exactly who do I act on first — and what
intervention do I use on each one — to save the most money?**

That's the question I set out to answer. Come with me through how it works — I'll keep
the jargon out of it, whether you came here for the business side or the data side.

---

## Part 1: Teaching a Model to Smell Trouble

Before you can decide *who to act on*, you have to know *who's at risk*. So that's where
I started — the data science half of the story.

I took a real dataset of credit-card members — their age, how long they'd been
with the bank, how much they spent, how often they transacted, how much of their
recent activity was slowing down. Buried in that behavior are the early warning
signs of someone quietly checking out: a customer whose transactions are thinning,
who's gone quiet for a couple of months, whose spending dropped off a cliff between
last quarter and this one.

I trained a machine-learning model (a random forest, running on PySpark so it could
scale) to learn those patterns and output a single number for each member: their
**probability of churning**.

And here's the first place I had to let business sense override the textbook — the kind
of trap most churn projects fall straight into. Only about 16% of members actually
churn. So a lazy model could just predict "nobody leaves" and be right 84% of the
time. Beautiful accuracy score. Utterly useless — because it never catches a single
person who's actually leaving.

So I threw out accuracy and optimized for **recall** — how many of the real churners
the model actually *catches*. After handling the imbalance, it catches **90% of the
people who really leave**, at a ROC-AUC of 0.98. That's the number that matters, and
it's the one most people forget to chase.

And the single biggest warning sign the model found? Not age, not income — it was
**transaction velocity**. When someone's number of transactions starts sliding,
they're already halfway out the door.

That's the data foundation. But a risk score on its own doesn't save anyone. Knowing
someone is 90% likely to leave doesn't tell you whether to call them, email them,
or leave them alone. That's where the business strategy begins.

---

## Part 2: Not Every At-Risk Customer Needs the Same Effort

Here's the trap most churn projects fall into: they rank everyone by risk and act on
the highest numbers. That sounds right and is quietly wrong.

Think about two customers, both at 90% risk of leaving:

- **Customer A** spends heavily, holds three products with the bank, and their
  activity is softening but they're still around and reachable.
- **Customer B** has a small, low-activity account, has gone completely quiet for four
  months, and hasn't logged in since spring.

They have the *same risk score*. But Customer A is worth fighting for and reachable.
Customer B is gone — calling them is throwing good money after bad. Risk alone can't
tell them apart.

So the engine weighs **two independent things** for every member:

- **Risk** — how likely are they to leave? (from the model)
- **Value** — how much are they worth? (their spending, their tenure, how many
  products they hold — a proxy for lifetime value)

And then it asks a smarter question than "who's most at risk?" It asks **"who's most
at risk *among the people an intervention can realistically reach and where the value
justifies the effort*?"** — and it lets the data decide
how much weight to give risk versus value, rather than me guessing. The optimizer
tries every possible balance and lands on the one that protects the most money while
still reaching enough at-risk people. It settles around a roughly even split — which
is itself a finding: neither risk nor value alone is enough.

---

## Part 3: The "Sleeping Dogs" and the Money-Waste Guard

Two more pieces of real-world judgment make this feel less like a math exercise and
more like something a retention leader would actually trust.

**Sleeping dogs.** These are the Customer B's of the world — high risk, but already
disengaged past the point of return. The engine identifies them not by their risk
score (which we've established is a poor signal for "hopeless") but by their
*behavior*: long dormancy plus collapsing activity. Once flagged, they're set aside.
Spending on them has negative return, so the system quietly refuses to.

**The money-waste guard.** An action isn't a guaranteed save — calling someone
doesn't magically retain them. So every member also gets a **save-probability**,
built purely from real behavior: are they still reachable? Is their recent activity
holding up? A disengaged member scores low here, which drags their expected return
down, so the math itself declines to spend on them. No hand-wavy rules — the
economics do the filtering.

Crucially, this does *not* mean writing off high-risk customers. A reachable member
at 95% risk is exactly who you *should* call — they're on the edge and still within
reach. The engine writes off the *unreachable*, not the *risky*. That distinction is
the whole game.

---

## Part 4: The Punchline — One Action Per Person

Here's where it all comes together into something a manager can actually use.

For every single member, the engine compares three possible actions:

- a **senior retention call** with a generous fee waiver (expensive, most effective),
- a cheaper **rep call** with a smaller waiver, or
- a low-cost **automated email**.

For each option it calculates the expected net return — *value at stake × chance the
action works × how effective that channel is, minus what the action costs* — and
**picks the single action that makes the most money for that specific person.**

A high-value, reachable, at-risk member earns a senior call. A moderate one gets a
rep call. The vast majority — low risk or low value — get an email, because a
$40 intervention to protect a $30 relationship is a loss, and the engine knows it.

This is the step that turns a prediction into a decision — and it's the part a
business will actually pay for. The result isn't a vague "15% of customers may churn."
It's a ranked, ready-to-work **action plan**: *here are the exact members to prioritize,
here's the specific intervention for each one, and here's the dollar value you'll protect
by doing it.*

---

## Bringing It to Life: Two Dashboards

Numbers in a spreadsheet don't persuade anyone. So the project has two front ends,
each for a different audience.

An **interactive Streamlit app** is the live decision tool. Slide the team's capacity
up or down, change the cost of an intervention, shift the emphasis between risk and value —
and watch the recommendations and the protected profit recompute in real time. It's
the engine's brain, exposed for someone to poke at and ask "what if?"

A polished **Tableau dashboard** is the executive view. The KPIs up top, a breakdown
of *who* we're targeting, the intervention mix, the prioritized action list, and a scenario comparison —
with a single toggle to see how the whole strategy shifts if the business's margin
assumptions change. This is the version you put in front of a decision-maker.

Same brain, two windows into it: one for exploring, one for reporting. That split —
heavy computation in Python, clean reporting in a BI tool — is exactly how real
analytics teams work.

---

## Why This Project Matters (to Both Sides of the Room)

**If you're a data person,** the interesting part isn't the model — random forests
are old news. It's the discipline around it: refusing to be fooled by accuracy on an
imbalanced target, keeping risk and value as independent signals so they don't
secretly double-count, and letting an optimizer choose the trade-off instead of
hard-coding a guess.

**If you're a business person,** the interesting part isn't the code. It's that every
number traces back to a real decision a retention team makes: don't chase lost
causes, don't ignore your loyal high-spenders, don't spend $40 to save $30, and
match the *intensity* of your outreach to the *value* on the line.

And the honest part — the part I'd say in any room — is knowing its limits. The model
runs on real data, but a few pieces (like how well each channel actually retains people)
are reasonable assumptions rather than measured facts. The clearest next step is an
*uplift model* trained on real intervention outcomes — who we contacted, what we
offered, and whether they actually stayed — to replace those assumptions with evidence.
I built the framework so that upgrade drops right in. Knowing what you'd fix next, and
saying it out loud, is part of doing the work well.

---

## The One-Sentence Version

**This project turns a wall of customer data into a single, practical answer to the
only question a retention team really has — *who do we act on first, and what
intervention do we use?* — while making sure every dollar of effort earns its keep.**
