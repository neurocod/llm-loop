#include "BadgeTest.h"

#include "Badge.h"

#include <QImage>
#include <QFontMetrics>
#include <QPainter>
#include <cstdio>
#include <optional>

namespace {

constexpr int kBlockHeight = 30;  // Badge.cpp's kHeight, in logical px

// Which logical rows of the FIRST block's text column (x 28..64, Badge.cpp's
// kColumnLeft/kColumnWidth) carry white
// ink. That is how the layout is told apart without duplicating the paint
// code: one 18 px line crosses the block's middle, two lines leave the 2 px
// gap between them (rows 14..15, Badge.cpp's kTwoLineGap) blank with ink
// above and below.
QVector<bool> inkRows(const QImage& image) {
	QVector<bool> rows(kBlockHeight, false);
	const double dpr = image.devicePixelRatio();
	const int left = qRound(28 * dpr);
	const int right = qMin(image.width(), qRound(64 * dpr));
	for (int y = 0; y < image.height(); ++y) {
		for (int x = left; x < right; ++x) {
			const QColor c = image.pixelColor(x, y);
			if (c.alpha() > 200 && c.red() > 200 && c.green() > 200 && c.blue() > 200) {
				rows[qMin(kBlockHeight - 1, static_cast<int>(y / dpr))] = true;
				break;
			}
		}
	}
	return rows;
}

bool inkIn(const QVector<bool>& rows, int from, int to) {
	for (int y = from; y < to; ++y)
		if (rows[y])
			return true;
	return false;
}

// Two lines: nothing in the middle band, something in each half.
bool isTwoLine(const QImage& image) {
	const QVector<bool> rows = inkRows(image);
	return !inkIn(rows, 14, 16) && inkIn(rows, 0, 14) && inkIn(rows, 16, kBlockHeight);
}

bool isOneLine(const QImage& image) {
	return inkIn(inkRows(image), 14, 16);
}

// The two lines fill the block: ink reaches within 4 px of its top and bottom
// edges, which a font laid out by its line box (ascent/descent) did not.
bool fillsTheBlock(const QImage& image) {
	const QVector<bool> rows = inkRows(image);
	return inkIn(rows, 0, 5) && inkIn(rows, kBlockHeight - 5, kBlockHeight);
}

AgentGroup group(const char* type, int count, std::optional<int> weekLeft) {
	AgentGroup out{QString::fromLatin1(type), count, 0};
	out.weekLeft = weekLeft;
	return out;
}

}  // namespace

int runBadgeSelfTest(const QString& previewPath) {
	int failures = 0;
	auto check = [&](bool ok, const char* description) {
		printf("%s  %s\n", ok ? "PASS" : "FAIL", description);
		if (!ok)
			++failures;
	};
#ifdef AGENT_COUNTER_PROVIDER_ICONS
	check(!QPixmap(QStringLiteral(":/agents/claude.png")).isNull(), "Claude icon is embedded and decodes");
	check(!QPixmap(QStringLiteral(":/agents/codex.png")).isNull(), "Codex icon is embedded and decodes");
#else
	// No AGENT_COUNTER_ICON_DIR: nothing may be embedded, so the badge paints
	// its own marks (the "provider identity" check below tells them apart).
	check(QPixmap(QStringLiteral(":/agents/claude.png")).isNull()
	          && QPixmap(QStringLiteral(":/agents/codex.png")).isNull(),
	      "no provider icon is embedded; the neutral marks are painted");
#endif

	Badge badge;
	// Keep the native platform's fonts/DPI while preventing any desktop window.
	// Qt's generic offscreen plugin can silently render digits as missing glyphs.
	badge.setAttribute(Qt::WA_DontShowOnScreen);
	const QFontMetrics metrics(badge.font());
	bool hasGlyphs = true;
	for (const QChar glyph : QStringLiteral("0123456789+?%↑↓"))
		hasGlyphs = hasGlyphs && metrics.inFont(glyph);
	check(hasGlyphs, "the platform font contains every badge glyph");
	badge.setState({{QStringLiteral("claude"), 1, 0}});
	const int singleWidth = badge.width();
	const QImage claude = badge.grab().toImage();
	check(!badge.isHidden(), "one active provider shows the badge");
	badge.setState({{QStringLiteral("codex"), 1, 0}});
	const QImage codex = badge.grab().toImage();
	check(claude != codex && badge.width() == singleWidth,
	      "provider identity changes the rendered icon, not the block width");
	badge.setState({{QStringLiteral("claude"), 2, 0}, {QStringLiteral("codex"), 3, 120}});
	check(badge.width() > 2 * singleWidth, "mixed providers have two blocks with a gap");
	const QImage mixed = badge.grab().toImage();
	badge.setState({{QStringLiteral("claude"), 12, 120}, {QStringLiteral("codex"), 123, 600}});
	const QImage crowded = badge.grab().toImage();
	check(mixed.size() == crowded.size(), "large counts keep provider block geometry stable");
	badge.setState({{QStringLiteral("claude"), 0, 0}, {QStringLiteral("codex"), 1, 0}});
	check(badge.width() == singleWidth, "the empty provider block disappears");
	badge.setState({{QStringLiteral("unknown"), 1, 0}});
	const QImage unknown = badge.grab().toImage();
	check(unknown != codex && unknown != claude, "untyped senders have an unbranded marker");
	badge.setState({});
	check(badge.isHidden(), "no running agents hides the whole badge");

	// --- the weekly figure: shown whenever reported, no threshold
	badge.setState({group("claude", 2, std::nullopt)});
	const QImage plain = badge.grab().toImage();
	check(isOneLine(plain), "without a figure the count is one line across the middle");

	badge.setState({group("claude", 2, 24)});
	const QImage low = badge.grab().toImage();
	check(isTwoLine(low) && badge.width() == singleWidth,
	      "a figure draws count over percent, at the same block width");
	check(fillsTheBlock(low), "the two lines are sized by their ink and fill the block's height");
	badge.setState({group("claude", 2, 17)});
	check(badge.grab().toImage() != low, "the drawn percent is the reported one");
	check(weekFigureText(QStringLiteral("claude"), 24) == QStringLiteral("↑76%")
	          && weekFigureText(QStringLiteral("codex"), 24) == QStringLiteral("↓24%")
	          && weekFigureText(QStringLiteral("unknown"), 24) == QStringLiteral("24%"),
	      "Claude shows use counting up, Codex the remainder counting down, untyped the bare remainder");
	check(weekFigureText(QStringLiteral("claude"), 0) == QStringLiteral("↑100%")
	          && weekFigureText(QStringLiteral("claude"), 100) == QStringLiteral("↑0%"),
	      "Claude's use is 100 minus the remainder at both ends");
	badge.setState({group("codex", 2, 24)});
	const QImage lowCodex = badge.grab().toImage();
	check(isTwoLine(lowCodex) && fillsTheBlock(lowCodex), "a Codex remainder draws the same two lines");
	badge.setState({group("claude", 2, 0)});
	check(isTwoLine(badge.grab().toImage()), "0 % left shows");
	badge.setState({group("claude", 2, 100)});
	check(isTwoLine(badge.grab().toImage()), "100 % left shows too: there is no threshold");
	badge.setState({group("claude", 123, 100), group("codex", 3, std::nullopt)});
	const QImage lowMixed = badge.grab().toImage();
	check(badge.width() == 2 * singleWidth + 4,
	      "a two-line block and a plain one keep both block widths");
	badge.setState({});

	if (!previewPath.isEmpty()) {
		// Render the real widget at 3x on both desktop backgrounds; do not
		// duplicate its paint logic in a mock-up that could hide layout defects.
		const QVector<QImage> samples{claude, codex, mixed, crowded, unknown, low, lowCodex, lowMixed};
		// Columns fit two 68 px blocks and their gap at 3x (420 px) plus margins.
		constexpr int kSheetColumn = 460;
		QImage sheet(2 * kSheetColumn, 1025, QImage::Format_ARGB32_Premultiplied);
		sheet.fill(QColor(35, 35, 35));
		QPainter painter(&sheet);
		painter.fillRect(kSheetColumn, 0, kSheetColumn, sheet.height(), QColor(240, 240, 240));
		for (int column = 0; column < 2; ++column) {
			int y = 20;
			for (const QImage& sample : samples) {
				const QSize logical = sample.deviceIndependentSize().toSize();
				painter.drawImage(QRect(QPoint(column * kSheetColumn + 15, y), logical * 3), sample);
				y += 125;
			}
		}
		painter.end();
		check(sheet.save(previewPath), "badge preview image saved");
	}
	return failures == 0 ? 0 : 1;
}
