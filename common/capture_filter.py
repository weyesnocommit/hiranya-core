import discord
import re

BAD_WORDS = {
	re.compile(r'([🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿]{2,})'): "oky"
}

class MessageFilter(object):
	@staticmethod
	def filter_content(message: discord.Message):
		filtered_content = message.content
		for bad_word_pattern, replacement in BAD_WORDS.items():
			filtered_content = bad_word_pattern.sub(replacement, filtered_content)
		return filtered_content
