// ---------------------------------------------------------------------------
// One icon + count block per running provider, at the top centre of the screen;
// the provider's weekly-limit figure joins the count as a second line.
// The window is translucent, click-through and hidden while no agents run.
// ---------------------------------------------------------------------------
#pragma once

#include <QWidget>
#include <QPixmap>

#include "Registry.h"

// A provider block shows its weekly figure under the count WHENEVER the
// provider reported one — no threshold (user decision 2026-11-06, replacing
// the first "below 25 %"). A block without a figure keeps the one-line look.
bool showsWeekLeft(const AgentGroup& group);

// The weekly line as each provider's own site states it (user, 2026-11-06):
// Codex counts the REMAINDER down ("↓76%"), Claude counts USE up
// ("↑24%" = 100 − remainder). The wire carries the remainder for both
// (`week_left`); an untyped sender gets the bare remainder, no arrow.
QString weekFigureText(const QString& agentType, int weekLeft);

// The cores block's count column: "leased/total".
QString coresText(const CoreSummary& cores);

class Badge : public QWidget {
	Q_OBJECT
public:
	Badge();

	// `oldestSeconds` is not decoration: the window is click-through, so there
	// is no hover and no tooltip, and colour is the only channel left to tell
	// "an agent is working" from "an agent has been stuck since this morning".
	//
	// `cores`, when shown, adds one block after the providers: "CPU" in the
	// icon slot and "leased/total" in the count column. Its colour is the
	// broker's state, not an age: slate while every request is served, amber
	// while anyone waits, red once forced grants put the leases over the
	// machine (CoreBroker::settle).
	void setState(const QVector<AgentGroup>& groups, const CoreSummary& cores = {});

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	void reposition();

	QVector<AgentGroup> m_groups;
	CoreSummary m_cores;
	QPixmap m_claudeIcon;
	QPixmap m_codexIcon;
};
