
import random
from typing import List
import re
import ast
from treebeard.store import Document
from treebeard.mcts import MCTSNode, ActionType
from evaluate import load
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

bertscore = load("bertscore")

class KnowledgeGap:
    def __init__(self, topic: str, context: str, importance: float = 1.0):
        self.topic = topic
        self.context = context
        self.importance = importance
        self.filled = False
        self.documents = []
        
    def add_document(self, doc: Document):
        self.documents.append(doc)
        if len(self.documents) >= 2:  # Simple heuristic - gap filled with 2+ docs
            self.filled = True
            
    def to_dict(self):
        return {
            "topic": self.topic,
            "filled": self.filled,
            "sources": [doc.id for doc in self.documents]
        }

class ReportSection:
    def __init__(self, title: str, content: str = ""):
        self.title = title
        self.content = content
        self.subsections = []
        self.documents = []  # Referenced documents
        self.knowledge_gaps = []
        self.confidence = 0.0
        self.gaps_identified = False
        self.content_generated = []
        
    def add_subsection(self, section):
        self.subsections.append(section)
        
    def add_knowledge_gap(self, gap: KnowledgeGap):
        self.knowledge_gaps.append(gap)
        
    def update_confidence(self):
        if not self.content:
            self.confidence = 0.0
        else:        
            results = bertscore.compute(predictions=[self.title], references=[self.content], lang="en")
            self.confidence = results['precision'][0]
        
    def has_unfilled_gaps(self):
        return any(not gap.filled for gap in self.knowledge_gaps)
        
    def to_dict(self):
        return {
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "subsections": [s.to_dict() for s in self.subsections],
            "doc_ids": [doc.id for doc in self.documents],
            "gaps": [gap.to_dict() for gap in self.knowledge_gaps],
        }
    
class ReportOutline:
    def __init__(self, title: str, sections: List[ReportSection] = None):
        self.title = title
        self.sections = sections or []
        self.overall_confidence = 0.0
        self.citation_count = 0
        
    def add_section(self, section: ReportSection):
        self.sections.append(section)
        
    def update_confidence(self):
        if not self.sections:
            self.overall_confidence = 0.0
        else:
            self.overall_confidence = sum(s.confidence for s in self.sections) / len(self.sections)
            
    def update_citation_count(self):
        self.citation_count = sum(len(s.documents) for s in self.sections)
        
    def has_unfilled_gaps(self):
        return any(s.has_unfilled_gaps() for s in self.sections)
        
    def to_dict(self):
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "confidence": self.overall_confidence,
            "citations": self.citation_count
        }
    
class MCTSDeepResearch:
    def __init__(self, vector_index, web_search, text_generator, 
                 embedder, max_iterations=100, max_depth=12):
        self.vector_index = vector_index
        self.web_search = web_search
        self.text_generator = text_generator
        self.embedder = embedder
        self.max_iterations = max_iterations
        self.search_budget = 2
        self.max_depth = max_depth
        

    def identify_main_topics(self, query, num_topics=5, outlineFromArchive=False):
        """Identify main topics for research using document clustering"""
        if not outlineFromArchive:
            topic_prompt = f"""Generate a simple outline for a research report on: {query}. 
             Outline MUST have only one level. You MUST return a valid Python list without any preamble.
             Outline MUST NOT have more than {num_topics} items.
             """
            topics_text = self.text_generator(topic_prompt)
            pattern = r'\[(?:\s*"[^"]*",?\s*)+\]'
            matches = re.findall(pattern, topics_text)

            outline = ast.literal_eval(matches[-1])

            topics = [f"{query}: {topic}" for topic in outline]
            print(topics)
            return topics

        documents = self.vector_index.documents
        embeddings = np.vstack([doc.embedding for doc in documents])

        max_clusters = min(num_topics, len(documents) - 1)
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, max_iter=5)
        clusters = kmeans.fit_predict(embeddings)

        keywords = [str(doc.keywords) for doc in documents]
        
        cluster_docs = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_docs:
                cluster_docs[cluster_id] = []
            cluster_docs[cluster_id].extend([keyword.strip() for keyword in keywords[i].split(",")])

        topics = []
        for cluster_id, docs in cluster_docs.items():

            word_counts = Counter(docs)
            
            for word in ENGLISH_STOP_WORDS:
                if word in word_counts:
                    del word_counts[word]

            top_words = [word for word, _ in word_counts.most_common(1)]

            if top_words:
                topic_name = f"{query}: {' '.join(top_words)}"
            else:
                topic_name = f"{query} cluster {cluster_id+1}"

            topics.append(topic_name)

        print(topics)
        return topics
    
    def generate_initial_outline(self, query, outlineFromArchive=False):
        """Generate initial report outline"""
        topics = self.identify_main_topics(query, outlineFromArchive=outlineFromArchive)
        
        outline = ReportOutline(f"Research Report: {query}")
        for topic in topics:
            outline.add_section(ReportSection(topic))
            
        return outline
    
    def get_available_actions(self, state):
        """Get available actions for the current state"""
        actions = []
        outline = state["outline"]
        current_section_idx = state.get("current_section_idx")
        searches_used = state.get("searches_used", 0)
        
        # Top-level actions
        if current_section_idx is None:
            # Select section to research/expand
            for i, _ in enumerate(outline.sections):
                actions.append((ActionType.SELECT_SECTION, i))

            # Finalize report if confidence is high enough
            if outline.overall_confidence > 0.7 and not outline.has_unfilled_gaps():
                actions.append((ActionType.FINALIZE, None))
        
        # Section-level actions
        else:
            section = outline.sections[current_section_idx]
            
            # Identify knowledge gaps
            if not section.knowledge_gaps:
                actions.append((ActionType.INITIAL_QUERY, current_section_idx))                        
                
            # Research actions for unfilled gaps
            unfilled_gaps = [gap for gap in section.knowledge_gaps if not gap.filled]

            if unfilled_gaps:
                # Vector search is always available
                actions.append((ActionType.VECTOR_SEARCH, current_section_idx))
                
                if searches_used < self.search_budget:
                    actions.append((ActionType.WEB_SEARCH, current_section_idx))
            
            # Generate content if we have docs
            if section.documents and not section.content:
                actions.append((ActionType.GENERATE_CONTENT, current_section_idx))
            
            if section.content and not section.gaps_identified:
                actions.append((ActionType.IDENTIFY_GAP, current_section_idx))
                
            # Integrate findings if we have new docs since last content generation
            if section.gaps_identified and any(gap.filled for gap in section.knowledge_gaps):
                actions.append((ActionType.INTEGRATE_FINDINGS, current_section_idx))
            
            # Return to top level
            actions.append((ActionType.BACK_TO_OUTLINE, None))

        return [(action_type, action_value) for action_type, action_value in actions]
    
    def apply_action(self, state, action_type, action_value):
        """Apply action to current state and return new state"""
        new_state = state.copy()
        outline = new_state["outline"]
        query = new_state["query"]
        searches_used = new_state.get("searches_used", 0)        
            
        if action_type == ActionType.SELECT_SECTION:
            new_state["current_section_idx"] = action_value            
            
        # Research actions
        elif action_type == ActionType.INITIAL_QUERY:
            section = outline.sections[action_value]
            section.add_knowledge_gap(KnowledgeGap(section.title, section.title))
                
        elif action_type == ActionType.IDENTIFY_GAP:
            section = outline.sections[action_value]
            research_questions_prompt = f"""
            Below is the content for this title: {section.title}
            Current content:
                {section.content}

            Identify any gaps in the suitability of the content for the title. 
            If such gaps exists, generate upto 3 questions that can be asked to fill the gap. 
            You MUST return a valid Python list without any preamble.
            """
            # print(research_questions_prompt)
            research_questions_text = self.text_generator(research_questions_prompt)
            # print(research_questions_text)
            pattern = r'\[(?:\s*"[^"]*",?\s*)+\]'
            matches = re.findall(pattern, research_questions_text)

            research_questions = ast.literal_eval(matches[-1])

            # Parse gaps (simplified)
            for question in research_questions:
                section.add_knowledge_gap(KnowledgeGap(question, section.title))
                
            section.gaps_identified = True
            
        elif action_type == ActionType.VECTOR_SEARCH:
            section = outline.sections[action_value]
            # Choose an unfilled gap
            unfilled_gaps = [gap for gap in section.knowledge_gaps if not gap.filled]
            if unfilled_gaps:
                gap = random.choice(unfilled_gaps)
                # Search for documents
                query_embedding = self.embedder(gap.topic)
                docs = self.vector_index.search(query_embedding)
                # Add results to the gap and section
                for doc in docs:
                    gap.add_document(doc)
                    if doc not in section.documents:
                        section.documents.append(doc)            
            
        elif action_type == ActionType.WEB_SEARCH:
            section = outline.sections[action_value]
            # Choose an unfilled gap
            unfilled_gaps = [gap for gap in section.knowledge_gaps if not gap.filled]
            if unfilled_gaps:
                gap = random.choice(unfilled_gaps)
                # Generate search query
                # search_prompt = f"Generate a simple web search query for: {gap.topic}. Do not inlcude any preamble."
                search_query = f"{outline.title} - {gap.topic}"#self.text_generator(search_prompt)

                # Perform search
                docs = self.web_search.search(search_query)
                # Add results
                for doc in docs:
                    gap.add_document(doc)
                    if doc not in section.documents:
                        section.documents.append(doc)
                new_state["searches_used"] = searches_used + 1
            
        elif action_type == ActionType.GENERATE_CONTENT:
            section = outline.sections[action_value]
            doc_contents = "\n".join([doc.content for doc in section.documents])
            filled_gaps = [gap.topic for gap in section.knowledge_gaps if gap.filled and gap.topic not in section.content_generated]
            
            section.content_generated.extend(filled_gaps)
            
            # Generate content with citations
            content_prompt = f"""Write a detailed and factual report on '{section.title}' addressing these topics:
            {', '.join(filled_gaps)}
            
            Based on ONLY these sources:
            {doc_contents}
            
            Do not include headings or titles.
            """
            
            section.content = self.text_generator(content_prompt)
            section.update_confidence()
            
        elif action_type == ActionType.INTEGRATE_FINDINGS:
            section = outline.sections[action_value]
            newly_filled = [gap for gap in section.knowledge_gaps if gap.filled and gap.topic not in section.content_generated]
            if newly_filled:
                new_docs = []
                for gap in newly_filled:
                    new_docs.extend(gap.documents)
                    section.content_generated.append(gap.topic)
                
                doc_contents = "\n".join([doc.content for doc in new_docs])
                integrate_prompt = f"""Incoporate the new information into the detailed report below:
                
                Current content:
                {section.content}
                
                New information on: {', '.join(gap.topic for gap in newly_filled)}
                Sources:
                {doc_contents}
                
                Do not include headings or titles.
                """
                
                section.content = self.text_generator(integrate_prompt)
                section.update_confidence()
            
        # Navigation actions
        elif action_type == ActionType.BACK_TO_OUTLINE:
            if "current_section_idx" in new_state:
                del new_state["current_section_idx"]
                
        elif action_type == ActionType.FINALIZE:
            new_state["completed"] = True
        
        # Update confidence metrics
        for section in outline.sections:
            section.update_confidence()
        outline.update_confidence()
        outline.update_citation_count()
            
        return new_state
    
    def select_node(self, node):
        """Select node using UCB1"""
        if not node.children:
            return node
            
        if not all(child.visits > 0 for child in node.children):
            unvisited = [c for c in node.children if c.visits == 0]
            return random.choice(unvisited)
            
        return max(node.children, key=lambda c: c.get_ucb_score())
    
    def expand(self, node):
        """Expand node by adding child nodes for all possible actions"""
        if node.available_actions is None:
            node.available_actions = self.get_available_actions(node.state)
            
        if node.available_actions:
            action_type, action_value = node.available_actions.pop(0)
            new_state = self.apply_action(node.state, action_type, action_value)
            return node.add_child(new_state, (action_type, action_value))
        return node
    
    def simulate(self, node):
        """Simulate from current node to terminal state"""
        if node.state.get("completed", False):
            # Terminal state reached - evaluate report quality
            return self.evaluate_report(node.state["outline"])
            
        # Random simulation
        state = node.state.copy()
        depth = 0
        
        while not state.get("completed", False) and depth < self.max_depth:
            actions = self.get_available_actions(state)
            if not actions:
                break
                
            action_type, action_value = random.choice(actions)
            state = self.apply_action(state, action_type, action_value)
            depth += 1
            
        # If reached completion or max depth, evaluate
        return self.evaluate_report(state["outline"]) * (0.95 ** depth)  # Discount for depth
    
    def backpropagate(self, node, reward):
        """Update node values up the tree"""
        while node:
            node.update(reward)
            node = node.parent
    
    def evaluate_report(self, outline):
        """Evaluate report quality"""
        # Coverage score
        sections_filled = sum(1 for s in outline.sections if s.content)
        coverage = sections_filled / max(1, len(outline.sections))
        
        # Confidence score
        confidence = outline.overall_confidence
        
        # Gap filling score
        gaps_filled = sum(1 for s in outline.sections for g in s.knowledge_gaps if g.filled)
        total_gaps = sum(len(s.knowledge_gaps) for s in outline.sections)
        gap_score = gaps_filled / max(1, total_gaps)
        
        # Citation score (normalized to prefer 3-5 citations per section)
        citation_ratio = outline.citation_count / max(1, len(outline.sections))
        citation_score = min(1.0, citation_ratio / 4)  # Optimal at 4 citations per section
        
        # Weighted combination
        return (coverage * 0.3) + (confidence * 0.3) + (gap_score * 0.3) + (citation_score * 0.1)
    
    def generate_research_report(self, query, outlineFromArchive=False):
        """Main method to generate a research report using MCTS"""
        # Generate initial outline
        initial_outline = self.generate_initial_outline(query, outlineFromArchive)
        
        initial_state = {
            "outline": initial_outline,
            "query": query,
            "searches_used": 0
        }
        
        # Create root node
        root = MCTSNode(initial_state)
        
        # Run MCTS
        for _ in range(self.max_iterations):
            # Selection
            node = root
            while node.children and node.is_fully_expanded():
                node = self.select_node(node)
                
            # Expansion
            if not node.state.get("completed", False):
                node = self.expand(node)
                
            # Simulation
            reward = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, reward)
            
            # Check if we have a completed report with good quality
            if root.value / max(1, root.visits) > 0.8:
                break
        
        # Find best report path
        current = root
        report_path = []
        
        while current.children:
            current = max(current.children, key=lambda c: c.value / max(1, c.visits))
            if current.action:
                report_path.append(current.action)
            
            if current.state.get("completed", False):
                break
        
        # Generate final report
        final_outline = current.state["outline"]
        searches_used = current.state.get("searches_used", 0)
        
        # Make sure all sections have content
        # for section in final_outline.sections:
        #     if not section.content and section.documents:
        #         doc_contents = "\n".join([doc.content for doc in section.documents])
        #         section.content = self.generator(f"Expand on '{section.title}' based on: {doc_contents}")
        
        return final_outline, report_path, searches_used
    
    def generate_final_document(self, outline):
        """Generate a complete formatted document from the outline"""
        title = outline.title
        sections = []
        
        for section in outline.sections:
            section_text = f"## {section.title}\n\n{section.content}\n\n"

            if section.documents:
                section_text += "### Sources\n"
                for doc in section.documents:
                    section_text += f"- {doc.get_citation()}\n"
            sections.append(section_text)
            
        document = f"# {title}\n\n"
        document += "\n".join(sections)
        
        # Add metadata
        document += f"\n\n---\n"
        document += f"Confidence Score: {outline.overall_confidence:.2f}\n"
        document += f"Citations: {outline.citation_count}\n"
        
        return document